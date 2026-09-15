#include <xpool/devkit/transport_observer.hpp>

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda/std/span>

#include <xpool/arena.hpp>
#include <xpool/debug/options.cuh>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/adapters.hpp>
#include <xpool/devkit/transport_observer.cuh>
#include <xpool/hooks.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/hooks.hpp>
#include <xpool/utils/layout.hpp>
#include <xpool/utils/trace.cuh>

namespace xpool::devkit::transport_observer {

namespace {

struct EndpointStorage {
  std::size_t instance_index = 0;
  std::size_t instance_rank = 0;
  xpool::utils::trace::BufferState *state = nullptr;
  Record *records = nullptr;
  std::size_t record_capacity = 0;
  std::uint64_t *current_sequence = nullptr;

  XPOOL_DEVICE_FN xpool::utils::trace::BufferView<Record> buffer() const {
    return {cuda::std::span{records, record_capacity}, *state};
  }
};

struct RegistryStorage {
  EndpointStorage *endpoints = nullptr;
  std::size_t endpoint_count = 0;
};

XPOOL_DEVICE_CONST RegistryStorage storage{};

struct Endpoint {
  xpool::hooks::TransportEndpointSite site;
  c10::DeviceIndex cuda_device;
  std::uint8_t *allocation;
  EndpointStorage storage;
};

std::mutex state_mutex;
std::vector<Endpoint> endpoints;
EndpointStorage *registry_allocation = nullptr;

void rebuild_registry() {
  // Endpoint open and close run outside resident-kernel service, so replacing
  // the device registry cannot race an endpoint lookup.
  if (registry_allocation != nullptr) {
    C10_CUDA_CHECK(cudaFree(registry_allocation));
    registry_allocation = nullptr;
  }
  auto entries = std::vector<EndpointStorage>{};
  entries.reserve(endpoints.size());
  for (const auto &endpoint : endpoints) {
    entries.push_back(endpoint.storage);
  }
  if (!entries.empty()) {
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&registry_allocation), entries.size() * sizeof(entries[0])));
    C10_CUDA_CHECK(
        cudaMemcpy(registry_allocation, entries.data(), entries.size() * sizeof(entries[0]), cudaMemcpyHostToDevice));
  }
  const auto registry = RegistryStorage{
      .endpoints = registry_allocation,
      .endpoint_count = entries.size(),
  };
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &registry, sizeof(registry)));
}

} // namespace

XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointOpenPostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().transport_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  const auto device_guard = c10::cuda::CUDAGuard{context.cuda_device};
  using xpool::utils::layout::LayoutRegionSpec;
  const auto specs = std::to_array<LayoutRegionSpec>({
      LayoutRegionSpec::object<xpool::utils::trace::BufferState>("Transport Observer state"),
      LayoutRegionSpec::array<Record>("Transport Observer records",
                                      xpool::debug::options().transport_observer.record_capacity),
      LayoutRegionSpec::object<std::uint64_t>("Transport Observer current sequence"),
  });
  const auto plan = xpool::utils::layout::LayoutPlan{specs, xpool::arena::kAllocationAlignment};
  auto index = std::size_t{0};
  const auto &state = plan[index++];
  const auto &records = plan[index++];
  const auto &current_sequence = plan[index++];
  TORCH_CHECK(index == plan.regions.size(), "xpool Transport Observer region plan is incomplete");
  auto *allocation = static_cast<std::uint8_t *>(nullptr);
  C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&allocation), plan.total_bytes));
  C10_CUDA_CHECK(cudaMemset(allocation, 0, plan.total_bytes));
  const auto endpoint_storage = EndpointStorage{
      .instance_index = context.layout.instance_index,
      .instance_rank = context.layout.instance_rank,
      .state = reinterpret_cast<xpool::utils::trace::BufferState *>(allocation + state.offset),
      .records = reinterpret_cast<Record *>(allocation + records.offset),
      .record_capacity = xpool::debug::options().transport_observer.record_capacity,
      .current_sequence = reinterpret_cast<std::uint64_t *>(allocation + current_sequence.offset),
  };
  endpoints.push_back(Endpoint{
      .site = context.site,
      .cuda_device = context.cuda_device,
      .allocation = allocation,
      .storage = endpoint_storage,
  });
  rebuild_registry();
}

XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointClosePreEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().transport_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  const auto device_guard = c10::cuda::CUDAGuard{context.cuda_device};
  const auto iterator = std::ranges::find_if(endpoints, [&](const auto &endpoint) {
    return endpoint.site == context.site && endpoint.storage.instance_index == context.layout.instance_index &&
           endpoint.storage.instance_rank == context.layout.instance_rank;
  });
  TORCH_CHECK(iterator != endpoints.end(), "xpool Transport Observer endpoint is not open");
  auto *allocation = iterator->allocation;
  endpoints.erase(iterator);
  rebuild_registry();
  C10_CUDA_CHECK(cudaFree(allocation));
}

namespace {

XPOOL_DEVICE_FN EndpointStorage *endpoint(const xpool::transport::ArenaView &arena) {
  const auto &layout = arena.layout();
  for (auto index = std::size_t{0}; index < storage.endpoint_count; ++index) {
    auto &candidate = storage.endpoints[index];
    if (candidate.instance_index == layout.instance_index && candidate.instance_rank == layout.instance_rank) {
      return &candidate;
    }
  }
  return nullptr;
}

EndpointSnapshot snapshot(const Endpoint &endpoint) {
  const auto device_guard = c10::cuda::CUDAGuard{endpoint.cuda_device};
  auto state = xpool::utils::trace::BufferState{};
  C10_CUDA_CHECK(cudaMemcpy(&state, endpoint.storage.state, sizeof(state), cudaMemcpyDeviceToHost));
  auto records = std::vector<Record>(std::min<std::size_t>(state.sequence, endpoint.storage.record_capacity));
  if (!records.empty()) {
    C10_CUDA_CHECK(cudaMemcpy(records.data(), endpoint.storage.records, records.size() * sizeof(records[0]),
                              cudaMemcpyDeviceToHost));
  }
  return {
      .instance_index = endpoint.storage.instance_index,
      .instance_rank = endpoint.storage.instance_rank,
      .sequence = state.sequence,
      .dropped = state.dropped,
      .records = std::move(records),
  };
}

} // namespace

XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportInstanceProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().transport_observer.enable) {
    return;
  }
  auto *owner = endpoint(context.arena);
  if (owner == nullptr) {
    return;
  }
  if (context.kind == xpool::hooks::TransportProtocolEventKind::RequestStagingStarted) {
    xpool::abort_if(context.request == nullptr);
    const auto entry = owner->buffer().reserve();
    *owner->current_sequence = entry ? entry->sequence() : 0;
    if (entry) {
      entry->record().begin_instance(entry->sequence(), context.payload_rows, *context.request);
    }
    return;
  }
  // Each endpoint owns one mailbox request at a time; its current sequence
  // therefore identifies the complete request-to-acknowledgement timeline.
  auto *record = owner->buffer().find(*owner->current_sequence);
  if (record == nullptr) {
    return;
  }
  switch (context.kind) {
  case xpool::hooks::TransportProtocolEventKind::RequestStagingCompleted:
    record->request_staging_completed();
    return;
  case xpool::hooks::TransportProtocolEventKind::RequestPublished:
    record->request_published();
    return;
  case xpool::hooks::TransportProtocolEventKind::ResultObserved:
    record->result_observed(context.result_code);
    return;
  case xpool::hooks::TransportProtocolEventKind::OutputCopied:
    record->output_copied();
    return;
  case xpool::hooks::TransportProtocolEventKind::ResultAcknowledged:
    record->result_acknowledged();
    *owner->current_sequence = 0;
    return;
  case xpool::hooks::TransportProtocolEventKind::Closed:
    if (record->recorded(xpool::hooks::TransportProtocolEventKind::ResultObserved)) {
      record->closed();
    } else {
      record->closed(context.result_code);
    }
    *owner->current_sequence = 0;
    return;
  default:
    return;
  }
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportAtnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().transport_observer.enable) {
    return;
  }
  auto *owner = endpoint(context.arena);
  if (owner == nullptr) {
    return;
  }
  if (context.kind == xpool::hooks::TransportProtocolEventKind::RequestObserved) {
    xpool::abort_if(context.request == nullptr);
    const auto entry = owner->buffer().reserve();
    *owner->current_sequence = entry ? entry->sequence() : 0;
    if (entry) {
      entry->record().begin_atnagent(entry->sequence(), context.payload_rows, *context.request);
    }
    return;
  }
  auto *record = owner->buffer().find(*owner->current_sequence);
  if (record == nullptr) {
    return;
  }
  switch (context.kind) {
  case xpool::hooks::TransportProtocolEventKind::ExecutionStarted:
    record->execution_started();
    return;
  case xpool::hooks::TransportProtocolEventKind::ExecutionCompleted:
    record->execution_completed();
    return;
  case xpool::hooks::TransportProtocolEventKind::ResultPublished:
    record->result_published(context.result_code);
    *owner->current_sequence = 0;
    return;
  default:
    return;
  }
}

std::optional<Snapshot> read() {
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  if (endpoints.empty()) {
    return std::nullopt;
  }
  auto snapshots = std::vector<EndpointSnapshot>{};
  snapshots.reserve(endpoints.size());
  for (const auto &endpoint : endpoints) {
    snapshots.push_back(snapshot(endpoint));
  }
  std::ranges::sort(snapshots, {},
                    [](const auto &value) { return std::pair{value.instance_index, value.instance_rank}; });
  return Snapshot{.endpoints = std::move(snapshots)};
}

} // namespace xpool::devkit::transport_observer
