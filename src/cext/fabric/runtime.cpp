#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/coordinator.hpp>
#include <xpool/fabric/ffnagent.hpp>
#include <xpool/fabric/hooks.hpp>
#include <xpool/fabric/module.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/ffnagent/hooks.hpp>
#include <xpool/hooks.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/hex.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::fabric {

namespace {

constexpr auto kResidentStartupTimeout = std::chrono::seconds{60};
constexpr auto kResidentStartupPollInterval = std::chrono::milliseconds{1};

} // namespace

Uid create_uid() {
  nvshmemx_uniqueid_t unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
  auto uid = Uid{unique_id};
  const auto status = nvshmemx_get_uniqueid(&uid.value());
  TORCH_CHECK(status == 0, "xpool failed to create an NVSHMEM unique id: ", status);
  return uid;
}

void Runtime::join(c10::DeviceIndex cuda_device, const ArenaProjection &projection, int pe) {
  TORCH_CHECK(cuda_device >= 0, "xpool Fabric join requires a non-negative CUDA device");
  TORCH_CHECK(pe >= 0 && static_cast<std::size_t>(pe) < projection.pe_count(),
              "xpool Fabric join PE index is out of range");

  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Empty, "xpool Fabric join requires an empty process runtime");
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device};
  try {
    // Phase: Materialize Layout - Project immutable Instance and Layer
    // topology into the typed tables and maximum storage geometry.
    drain_stream_ = xpool::utils::device::OwnedCudaStream::create();
    auto layer_count = std::size_t{0};
    for (const auto &instance : projection.instances) {
      layer_count = xpool::utils::checked::sum(layer_count, instance.layers.size());
    }

    auto instance_entries = std::vector<InstanceEntry>{};
    instance_entries.reserve(projection.instances.size());
    auto layer_entries = std::vector<LayerEntry>{};
    layer_entries.reserve(layer_count);
    auto atnagent_pes = std::vector<int>{};
    auto ffnagent_pes = std::vector<int>{};
    auto maximum_lane_payload_bytes = std::size_t{0};
    auto maximum_routing_metadata_elements = std::size_t{0};
    for (const auto &instance : projection.instances) {
      const auto maximum_rows = std::max(instance.decode_payload_row_capacity, instance.prefill_payload_row_capacity);
      const auto payload_row_bytes = xpool::utils::checked::prod(
          instance.hidden_size, static_cast<std::size_t>(c10::elementSize(instance.payload_dtype)));
      const auto payload_bytes = xpool::utils::checked::prod(maximum_rows, payload_row_bytes);
      maximum_lane_payload_bytes = std::max(maximum_lane_payload_bytes, payload_bytes);
      const auto ffn_tp_size = instance.layers.front().ffnagent_indices.size();
      instance_entries.push_back(InstanceEntry{
          .payload_dtype = instance.payload_dtype,
          .group_sum_complete_admitted = instance.group_sum_complete_admitted,
          .hidden_size = instance.hidden_size,
          .payload_row_bytes = payload_row_bytes,
          .decode_payload_row_capacity = instance.decode_payload_row_capacity,
          .prefill_payload_row_capacity = instance.prefill_payload_row_capacity,
          .atn_tp_size = instance.atn_tp_size,
          .atn_dp_size = instance.atn_dp_size,
          .layer_begin = layer_entries.size(),
          .layer_count = instance.layers.size(),
          .atnagent_pe_begin = atnagent_pes.size(),
          .ffnagent_pe_begin = ffnagent_pes.size(),
          .ffn_tp_size = ffn_tp_size,
      });
      for (const auto index : instance.atnagent_indices) {
        atnagent_pes.push_back(static_cast<int>(index));
      }
      for (const auto &layer : instance.layers) {
        layer_entries.push_back(LayerEntry{
            .layer_id = layer.layer_id,
            .kind = layer.kind,
            .effective_topk = layer.effective_topk,
        });
        for (const auto index : layer.ffnagent_indices) {
          ffnagent_pes.push_back(static_cast<int>(projection.atnagent_count + index));
        }
        if (layer.kind == xpool::ffn::LayerKind::Moe) {
          maximum_routing_metadata_elements = std::max(maximum_routing_metadata_elements,
                                                       xpool::utils::checked::prod(maximum_rows, layer.effective_topk));
        }
      }
    }

    const auto layout =
        ArenaLayout::create(projection.atnagent_count, projection.ffnagent_count, instance_entries.size(),
                                  projection.executor_lane_count, layer_entries.size(), atnagent_pes.size(),
                                  ffnagent_pes.size(), maximum_lane_payload_bytes, maximum_routing_metadata_elements);

    // Phase: Initialize NVSHMEM - Join the exact projected PE set and validate
    // the Device-visible runtime identity before allocating shared resources.
    const auto &unique_id = projection.uid.value();
    nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
    auto status = nvshmemx_set_attr_uniqueid_args(pe, static_cast<int>(projection.pe_count()), &unique_id, &attributes);
    TORCH_CHECK(status == 0, "xpool failed to configure NVSHMEM unique-id attributes: ", status);
    status = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attributes);
    TORCH_CHECK(status == 0, "xpool failed to initialize the NVSHMEM host library: ", status);
    const auto init_status = nvshmemx_init_status();
    TORCH_CHECK(init_status >= NVSHMEM_STATUS_IS_INITIALIZED && init_status < NVSHMEM_STATUS_INVALID,
                "xpool NVSHMEM device initialization is incomplete: ", init_status);
    TORCH_CHECK(nvshmem_my_pe() == pe && nvshmem_n_pes() == static_cast<int>(projection.pe_count()),
                "xpool NVSHMEM runtime identity differs from Fabric join metadata");

    // Phase: Establish Participant Resources - Module registration precedes
    // symmetric Arena creation and remains live through resident drain.
    auto module_registration = ModuleRegistration::create();
    auto arena = Arena::create(layout, std::span<const InstanceEntry>{instance_entries},
                                     std::span<const LayerEntry>{layer_entries},
                                     std::span<const int>{atnagent_pes}, std::span<const int>{ffnagent_pes});
    // No participant publishes Joined until every PE initialized the same
    // symmetric layout and reached this collective boundary.
    nvshmem_barrier_all();

    // Phase: Commit Runtime State - Publish Joined only after every resource is
    // complete, so an exception leaves no partially joined runtime visible.
    cuda_device_ = cuda_device;
    projection_ = projection;
    pe_ = pe;
    arena_ = std::move(arena);
    module_registration_ = std::move(module_registration);
    phase_ = Phase::Joined;
    xpool::hooks::FabricJoinPostEvent::hooks(
        {.cuda_device = cuda_device, .pe = pe, .layout = arena_.layout(), .projection = *projection_});
  } catch (...) {
    phase_ = Phase::Closed;
    throw;
  }
}

void Runtime::install_ffnagent_execution(const xpool::ffnagent::ExecutionProjection &projection) {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Joined, "xpool FFN execution install requires a joined process runtime");
  TORCH_CHECK(static_cast<std::size_t>(*pe_) >= projection_->atnagent_count,
              "xpool FFN execution install is valid only for an FfnAgent PE");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  if (!ffnagent_control_) {
    ffnagent_control_ = FfnAgentControl::create(*pe_ == projection_->coordinator_pe(), projection_->scheduler,
                                                projection_->instances.size(), projection_->executor_lane_count);
  }
  const auto ffnagent_index = static_cast<std::size_t>(*pe_) - projection_->atnagent_count;
  ffn_execution_runtime_.install(projection, arena_.view(), arena_.layout(), *projection_, ffnagent_index,
                                 ffnagent_control_.view().activation_count);
}

void Runtime::activate_ffnagent() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Joined, "xpool Fabric activate requires a joined process runtime");
  TORCH_CHECK(static_cast<std::size_t>(*pe_) >= projection_->atnagent_count,
              "xpool Fabric activate is valid only for an FfnAgent PE");
  TORCH_CHECK(!resident_stream_, "xpool FfnAgent Resident is already active");

  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  const auto is_coordinator = *pe_ == projection_->coordinator_pe();
  TORCH_CHECK(ffnagent_control_ && ffn_execution_runtime_.installed(),
              "xpool FfnAgent activation requires installed execution");
  const auto control = ffnagent_control_.view();
  if (is_coordinator) {
    resident_stream_ = xpool::utils::device::OwnedCudaStream::create();
    launch_coordinator(arena_.view(), arena_.layout(), control, resident_stream_.get());
  }
  ffn_execution_runtime_.activate();

  const auto expected_activation_count =
      static_cast<std::uint32_t>(projection_->executor_lane_count) + (is_coordinator ? 1U : 0U);
  const auto ready = [&] {
    auto observed = std::uint32_t{0};
    C10_CUDA_CHECK(cudaMemcpy(&observed, control.activation_count, sizeof(observed), cudaMemcpyDeviceToHost));
    TORCH_CHECK(observed <= expected_activation_count, "xpool FfnAgent Resident over-published activation count");
    return observed == expected_activation_count;
  };
  const auto result = xpool::utils::wait::until(
      std::chrono::steady_clock::now() + kResidentStartupTimeout, ready,
      [&] {
        if (resident_stream_ && resident_stream_.query()) {
          return true;
        }
        ffn_execution_runtime_.check_health();
        return false;
      },
      kResidentStartupPollInterval);
  switch (result) {
  case xpool::utils::wait::Status::Pending:
    break;
  case xpool::utils::wait::Status::Ready:
    return;
  case xpool::utils::wait::Status::Cancelled:
    TORCH_CHECK(false, "xpool FfnAgent Resident completed before every execution owner published activation");
  case xpool::utils::wait::Status::TimedOut:
    TORCH_CHECK(false, "xpool FfnAgent Resident startup exceeded the bounded deadline");
  }
  xpool::abort();
}

void Runtime::check_ffnagent_health() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Joined, "xpool Fabric health requires a joined process runtime");
  TORCH_CHECK(static_cast<std::size_t>(*pe_) >= projection_->atnagent_count,
              "xpool FfnAgent health is valid only for an FfnAgent PE");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  const auto state = arena_.state();
  if (state.failure.publication == 1) {
    return;
  }
  TORCH_CHECK(resident_stream_ || ffn_execution_runtime_.installed(), "xpool FfnAgent runtime has not been activated");
  ffn_execution_runtime_.check_health();
  TORCH_CHECK(!resident_stream_ || !resident_stream_.query(), "xpool Fabric Coordinator completed unexpectedly");
}

ArenaView Runtime::arena() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Joined || phase_ == Phase::Draining || phase_ == Phase::Drained,
              "xpool Fabric arena is unavailable outside a joined generation");
  return arena_.view();
}

void Runtime::drain_async() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  if (phase_ == Phase::Draining || phase_ == Phase::Drained) {
    return;
  }
  TORCH_CHECK(phase_ == Phase::Joined, "xpool Fabric drain requires a joined process runtime");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  // Shutdown publication is asynchronous so the Python control plane can keep
  // reporting progress while all PEs cooperatively retire their residents.
  arena_.request_shutdown(drain_stream_);
  phase_ = Phase::Draining;
}

bool Runtime::drain_pending() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  if (phase_ == Phase::Drained) {
    return false;
  }
  TORCH_CHECK(phase_ == Phase::Draining, "xpool Fabric drain has not been started");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  if (!drain_stream_.query() || (resident_stream_ && !resident_stream_.query()) ||
      (ffn_execution_runtime_.installed() && ffn_execution_runtime_.drain_pending())) {
    return true;
  }
  // Stream completion is the local proof that shutdown publication completed
  // and every resident kernel owned by this PE has exited.
  drain_stream_.destroy();
  resident_stream_.destroy();
  phase_ = Phase::Drained;
  return false;
}

std::optional<Failure> Runtime::failure() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Joined || phase_ == Phase::Draining || phase_ == Phase::Drained,
              "xpool Fabric failure is unavailable outside a joined generation");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  const auto state = arena_.state();
  if (state.failure.publication == 0) {
    return std::nullopt;
  }
  TORCH_CHECK(state.failure.publication == 1, "xpool Fabric failure has an invalid publication value");
  const auto result_code = state.failure.payload.result_code;
  TORCH_CHECK(result_code == xpool::ffn::ResultCode::ProtocolMismatch ||
                  result_code == xpool::ffn::ResultCode::Timeout,
              "xpool Fabric failure has an invalid result code");
  TORCH_CHECK(state.failure.payload.origin_pe >= 0 &&
                  static_cast<std::size_t>(state.failure.payload.origin_pe) < projection_->pe_count(),
              "xpool Fabric failure has an invalid origin PE");
  TORCH_CHECK(state.failure.payload.key.valid() &&
                  state.failure.payload.key.instance_index < projection_->instances.size(),
              "xpool Fabric failure has an invalid invocation key");
  TORCH_CHECK(state.failure.payload.layer_ordinal <
                  projection_->instances[state.failure.payload.key.instance_index].layers.size(),
              "xpool Fabric failure has an invalid layer ordinal");
  return state.failure;
}

void Runtime::finalize() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Drained, "xpool Fabric finalize requires completed local drain");
  phase_ = Phase::Closed;
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  // Teardown order is contractual: release symmetric allocations while
  // NVSHMEM and its CUDA module remain live, unregister the module, then
  // finalize the participant-local host library.
  if (ffn_execution_runtime_.installed()) {
    ffn_execution_runtime_.finalize();
  }
  xpool::hooks::FabricFinalizePreEvent::hooks({.cuda_device = *cuda_device_});
  ffnagent_control_.destroy();
  arena_.destroy();
  module_registration_.destroy();
  nvshmemx_hostlib_finalize();
  cuda_device_.reset();
  projection_.reset();
  pe_.reset();
}

} // namespace xpool::fabric
