#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/adapters.hpp>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cooperative_groups.h>
#include <cuda/std/span>

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>
#include <vector>

#include <xpool/arena.hpp>
#include <xpool/debug/options.cuh>
#include <xpool/fabric/hooks.hpp>
#include <xpool/devkit/fabric_observer.cuh>
#include <xpool/hooks.cuh>
#include <xpool/macros.hpp>
#include <xpool/utils/layout.hpp>
#include <xpool/utils/trace.cuh>

namespace xpool::devkit::fabric_observer {

namespace {

struct Storage {
  xpool::utils::trace::BufferState *state = nullptr;
  Record *records = nullptr;
  std::size_t record_capacity = 0;
  std::uint64_t *atnagent_trace_ids = nullptr;
  std::uint64_t *coordinator_trace_ids = nullptr;
  std::uint64_t *ffnagent_trace_ids = nullptr;

  XPOOL_DEVICE_FN xpool::utils::trace::BufferView<Record> buffer() const {
    return {cuda::std::span{records, record_capacity}, *state};
  }

  XPOOL_DEVICE_FN Record *find(std::uint64_t sequence) const {
    return buffer().find(sequence);
  }
};

// The Host installs this process-local pointer table only at Fabric join and
// clears it before finalization; resident kernels therefore observe one stable
// device-constant view for their entire service lifetime.
XPOOL_DEVICE_CONST Storage storage{};

auto storage_plan(std::size_t record_capacity, std::size_t instance_count, std::size_t executor_lane_count) {
  using xpool::utils::layout::LayoutRegionSpec;
  const auto specs = std::to_array<LayoutRegionSpec>({
      LayoutRegionSpec::object<xpool::utils::trace::BufferState>("Fabric Observer state"),
      LayoutRegionSpec::array<Record>("Fabric Observer records", record_capacity),
      LayoutRegionSpec::array<std::uint64_t>("Fabric AtnAgent trace ids", instance_count),
      LayoutRegionSpec::array<std::uint64_t>("Fabric Coordinator trace ids", instance_count),
      LayoutRegionSpec::array<std::uint64_t>("Fabric FfnAgent trace ids", executor_lane_count),
  });
  return xpool::utils::layout::LayoutPlan{specs, xpool::arena::kAllocationAlignment};
}

struct HostState {
  c10::DeviceIndex cuda_device;
  int pe;
  std::uint8_t *allocation;
  Storage storage;
  std::size_t atnagent_count;
  std::size_t ffnagent_count;
  std::vector<ModelTopology> model_topologies;
};

std::mutex state_mutex;
std::optional<HostState> host_state;

} // namespace

std::size_t allocation_bytes(std::size_t instance_count, std::size_t executor_lane_count) {
  TORCH_CHECK(instance_count != 0 && executor_lane_count != 0,
              "xpool Fabric Observer requires positive Instance and executor-Lane counts");
  const auto &observer = xpool::debug::options().fabric_observer;
  return observer.enable ? storage_plan(observer.record_capacity, instance_count, executor_lane_count).total_bytes : 0;
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FabricJoinPostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().fabric_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  TORCH_CHECK(!host_state.has_value(), "xpool Fabric Observer is already open");
  const auto device_guard = c10::cuda::CUDAGuard{context.cuda_device};
  const auto plan = storage_plan(xpool::debug::options().fabric_observer.record_capacity,
                                 context.layout.instance_count, context.layout.executor_lane_count);
  auto index = std::size_t{0};
  const auto &state = plan[index++];
  const auto &records = plan[index++];
  const auto &atnagent_trace_ids = plan[index++];
  const auto &coordinator_trace_ids = plan[index++];
  const auto &ffnagent_trace_ids = plan[index++];
  TORCH_CHECK(index == plan.regions.size(), "xpool Fabric Observer region plan is incomplete");
  auto *allocation = static_cast<std::uint8_t *>(nullptr);
  C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&allocation), plan.total_bytes));
  C10_CUDA_CHECK(cudaMemset(allocation, 0, plan.total_bytes));
  const auto observer_storage = Storage{
      .state = reinterpret_cast<xpool::utils::trace::BufferState *>(allocation + state.offset),
      .records = reinterpret_cast<Record *>(allocation + records.offset),
      .record_capacity = xpool::debug::options().fabric_observer.record_capacity,
      .atnagent_trace_ids = reinterpret_cast<std::uint64_t *>(allocation + atnagent_trace_ids.offset),
      .coordinator_trace_ids = reinterpret_cast<std::uint64_t *>(allocation + coordinator_trace_ids.offset),
      .ffnagent_trace_ids = reinterpret_cast<std::uint64_t *>(allocation + ffnagent_trace_ids.offset),
  };
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &observer_storage, sizeof(observer_storage)));
  auto model_topologies = std::vector<ModelTopology>{};
  model_topologies.reserve(context.projection.instances.size());
  for (const auto &instance : context.projection.instances) {
    model_topologies.push_back({.atn_tp_size = instance.atn_tp_size, .atn_dp_size = instance.atn_dp_size});
  }
  host_state.emplace(HostState{.cuda_device = context.cuda_device,
                               .pe = context.pe,
                               .allocation = allocation,
                               .storage = observer_storage,
                               .atnagent_count = context.layout.atnagent_count,
                               .ffnagent_count = context.layout.ffnagent_count,
                               .model_topologies = std::move(model_topologies)});
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FabricFinalizePreEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().fabric_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  const auto device_guard = c10::cuda::CUDAGuard{context.cuda_device};
  const auto empty = Storage{};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &empty, sizeof(empty)));
  C10_CUDA_CHECK(cudaFree(host_state->allocation));
  host_state.reset();
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricAtnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().fabric_observer.enable || storage.state == nullptr) {
    return;
  }
  // Model-internal requests are serialized, so one Instance-local sequence is
  // sufficient to correlate the AtnAgent events between begin and acknowledgement.
  auto *record = storage.find(storage.atnagent_trace_ids[context.instance_index]);
  switch (context.kind) {
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPrepared: {
    xpool::abort_if(context.submission == nullptr);
    const auto entry = storage.buffer().reserve();
    storage.atnagent_trace_ids[context.instance_index] = entry ? entry->sequence() : 0;
    if (entry) {
      entry->record().begin_atnagent(entry->sequence(), *context.submission);
    }
    return;
  }
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPublished:
    break;
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::AdmissionObserved:
    xpool::abort_if(context.admission == nullptr);
    record = storage.find(storage.atnagent_trace_ids[context.instance_index]);
    if (record != nullptr) {
      record->admission_observed(*context.admission);
    }
    return;
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::InputReadyPublished:
    break;
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputCommitObserved:
    xpool::abort_if(context.commit == nullptr);
    record = storage.find(storage.atnagent_trace_ids[context.instance_index]);
    if (record != nullptr) {
      record->output_commit_observed(*context.commit);
    }
    return;
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputAcknowledgementPublished:
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::Count:
    break;
  }
  if (record == nullptr) {
    return;
  }
  switch (context.kind) {
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPublished:
    record->submission_published();
    return;
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::InputReadyPublished:
    record->input_ready_published();
    return;
  case xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputAcknowledgementPublished:
    record->output_acknowledgement_published();
    storage.atnagent_trace_ids[context.instance_index] = 0;
    return;
  default:
    return;
  }
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricCoordinatorProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().fabric_observer.enable || storage.state == nullptr) {
    return;
  }
  auto *record = storage.find(storage.coordinator_trace_ids[context.instance_index]);
  switch (context.kind) {
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Enqueued: {
    xpool::abort_if(context.invocation == nullptr);
    const auto entry = storage.buffer().reserve();
    storage.coordinator_trace_ids[context.instance_index] = entry ? entry->sequence() : 0;
    if (entry) {
      entry->record().begin_coordinator(entry->sequence(), *context.invocation);
      entry->record().enqueued(context.ready_ticket);
    }
    return;
  }
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Scheduled:
    if (record != nullptr) {
      record->scheduled(context.executor_lane_index, context.executor_lease_sequence);
    }
    return;
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::AdmissionPublished:
    if (record != nullptr) {
      record->admission_published();
    }
    return;
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneExecutionPublished:
    if (record != nullptr) {
      record->lane_execution_published();
    }
    return;
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::FfnAgentCompletionsObserved:
    if (record != nullptr) {
      record->ffnagent_completions_observed();
    }
    return;
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputCommitPublished:
    if (record != nullptr) {
      record->output_commit_published();
    }
    return;
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputAcknowledgementsObserved:
    if (record != nullptr) {
      record->output_acknowledgements_observed();
    }
    return;
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneReleased:
    if (record != nullptr) {
      record->lane_released();
      storage.coordinator_trace_ids[context.instance_index] = 0;
    }
    return;
  case xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Count:
    return;
  }
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().fabric_observer.enable || storage.state == nullptr ||
      cooperative_groups::this_thread_block().thread_rank() != 0) {
    return;
  }
  // A Lane owns at most one lease at a time, so its sequence remains stable
  // until Completion closes the trace and permits Lane reuse.
  auto *record = storage.find(storage.ffnagent_trace_ids[context.executor_lane_index]);
  switch (context.kind) {
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::LaneExecutionObserved: {
    xpool::abort_if(context.execution == nullptr);
    const auto entry = storage.buffer().reserve();
    storage.ffnagent_trace_ids[context.executor_lane_index] = entry ? entry->sequence() : 0;
    if (entry) {
      entry->record().begin_ffnagent(entry->sequence(), *context.execution, context.executor_lane_index,
                                     context.payload_row_capacity, context.delivery);
    }
    return;
  }
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::InputReadyObserved:
    if (record != nullptr) {
      record->input_ready_observed();
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataPublished:
    if (record != nullptr) {
      record->routing_metadata_published();
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataObserved:
    if (record != nullptr) {
      record->routing_metadata_observed();
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeStarted:
    if (record != nullptr) {
      record->compute_started();
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeCompleted:
    if (record != nullptr) {
      record->compute_completed();
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PartialReadyPublished:
    if (record != nullptr) {
      record->partial_ready_published();
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PeerPartialsReadyObserved:
    if (record != nullptr) {
      record->peer_partials_ready_observed();
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::CompletionPublished:
    if (record != nullptr) {
      record->completion_published();
      storage.ffnagent_trace_ids[context.executor_lane_index] = 0;
    }
    return;
  case xpool::hooks::FabricFfnAgentProtocolEvent::Kind::Count:
    return;
  }
}

std::optional<Snapshot> read() {
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  if (!host_state.has_value()) {
    return std::nullopt;
  }
  const auto device_guard = c10::cuda::CUDAGuard{host_state->cuda_device};
  auto state = xpool::utils::trace::BufferState{};
  C10_CUDA_CHECK(cudaMemcpy(&state, host_state->storage.state, sizeof(state), cudaMemcpyDeviceToHost));
  const auto retained = std::min<std::size_t>(state.sequence, host_state->storage.record_capacity);
  auto records = std::vector<Record>(retained);
  if (!records.empty()) {
    C10_CUDA_CHECK(cudaMemcpy(records.data(), host_state->storage.records,
                              records.size() * sizeof(records.front()), cudaMemcpyDeviceToHost));
  }
  return Snapshot{
      .pe = host_state->pe,
      .atnagent_count = host_state->atnagent_count,
      .ffnagent_count = host_state->ffnagent_count,
      .model_topologies = host_state->model_topologies,
      .sequence = state.sequence,
      .dropped = state.dropped,
      .records = std::move(records),
  };
}

} // namespace xpool::devkit::fabric_observer
