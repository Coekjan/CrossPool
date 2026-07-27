#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <xpool/fabric/ffnagent.hpp>
#include <xpool/fabric/module.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/hex.hpp>

namespace xpool::fabric {

namespace {

constexpr auto kResidentStartupTimeout = std::chrono::seconds{60};

} // namespace

FabricUid create_uid() {
  nvshmemx_uniqueid_t unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
  auto uid = FabricUid{unique_id};
  const auto status = nvshmemx_get_uniqueid(&uid.value());
  TORCH_CHECK(status == 0, "xpool failed to create an NVSHMEM unique id: ", status);
  return uid;
}

void FabricJoinMetadata::validate() const {
  TORCH_CHECK(
      atnagent_count != 0 && ffnagent_count != 0,
      "xpool Fabric topology requires positive Agent counts");
  const auto participant_count = xpool::utils::checked::sum(atnagent_count, ffnagent_count);
  TORCH_CHECK(
      participant_count <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "xpool Fabric PE count exceeds the NVSHMEM integer domain");
  TORCH_CHECK(
      pe >= 0 && static_cast<std::size_t>(pe) < participant_count,
      "xpool Fabric join PE index is out of range");
  TORCH_CHECK(executor_count != 0, "xpool Fabric join requires at least one Executor");
  TORCH_CHECK(!models.empty(), "xpool Fabric join requires at least one model");
  static_cast<void>(scheduler_policy.type());

  auto total_layer_count = std::size_t{0};
  for (const auto &model : models) {
    TORCH_CHECK(
        model.max_decode_rows != 0 && model.max_prefill_rows != 0 &&
            model.hidden_size != 0 && model.atn_tp_size != 0 && model.atn_dp_size != 0,
        "xpool Fabric model metadata has zero geometry");
    TORCH_CHECK(
        xpool::utils::checked::prod(model.atn_tp_size, model.atn_dp_size) ==
            atnagent_count,
        "xpool Fabric model topology does not cover every AtnAgent");
    TORCH_CHECK(!model.layers.empty(), "xpool Fabric model requires at least one FFN layer");
    total_layer_count = xpool::utils::checked::sum(total_layer_count, model.layers.size());
    auto layer_ids = std::unordered_set<std::size_t>{};
    layer_ids.reserve(model.layers.size());
    for (const auto &layer : model.layers) {
      static_cast<void>(layer.kind.value());
      TORCH_CHECK(
          layer_ids.insert(layer.layer_id).second,
          "xpool Fabric model contains duplicate FFN layer ids");
    }
  }
  TORCH_CHECK(total_layer_count != 0, "xpool Fabric join requires at least one FFN layer");
}

void FabricRuntime::join(
    c10::DeviceIndex cuda_device,
    const FabricJoinMetadata &metadata) {
  TORCH_CHECK(cuda_device >= 0, "xpool Fabric join requires a non-negative CUDA device");
  metadata.validate();

  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Empty, "xpool Fabric join requires an empty process runtime");
  const auto device_guard = c10::cuda::CUDAGuard{cuda_device};
  try {
    auto layer_count = std::size_t{0};
    for (const auto &model : metadata.models) {
      layer_count = xpool::utils::checked::sum(layer_count, model.layers.size());
    }

    auto model_layouts = std::vector<FabricModelLayout>{};
    model_layouts.reserve(metadata.models.size());
    auto layer_layouts = std::vector<FabricLayerLayout>{};
    layer_layouts.reserve(layer_count);
    auto model_payloads_bytes = std::size_t{0};
    auto executor_payload_capacity_bytes = std::size_t{0};
    for (const auto &model : metadata.models) {
      const auto decode_bytes = xpool::utils::checked::prod(
          model.max_decode_rows, model.hidden_size, model.dtype.bytes());
      const auto prefill_bytes = xpool::utils::checked::prod(
          model.max_prefill_rows, model.hidden_size, model.dtype.bytes());
      const auto decode_capacity =
          xpool::utils::checked::align_up(decode_bytes,
                                          xpool::arena::kPayloadAlignment);
      const auto prefill_capacity =
          xpool::utils::checked::align_up(prefill_bytes,
                                          xpool::arena::kPayloadAlignment);
      const auto decode_offset = model_payloads_bytes;
      model_payloads_bytes =
          xpool::utils::checked::sum(model_payloads_bytes, decode_capacity);
      executor_payload_capacity_bytes =
          std::max(executor_payload_capacity_bytes, prefill_capacity);
      model_layouts.push_back(FabricModelLayout{
          .dtype = model.dtype.value(),
          .hidden_size = model.hidden_size,
          .atn_tp_size = model.atn_tp_size,
          .atn_dp_size = model.atn_dp_size,
          .layer_begin = layer_layouts.size(),
          .layer_count = model.layers.size(),
          .decode_payload_offset = decode_offset,
          .decode_payload_capacity_bytes = decode_capacity,
          .prefill_payload_capacity_bytes = prefill_capacity,
      });
      for (const auto &layer : model.layers) {
        layer_layouts.push_back(FabricLayerLayout{
            .layer_id = layer.layer_id,
            .kind = layer.kind.value(),
        });
      }
    }

    const auto layout = FabricArenaLayout::create(
        metadata.atnagent_count, metadata.ffnagent_count, metadata.executor_count,
        model_layouts.size(), layer_layouts.size(), model_payloads_bytes,
        executor_payload_capacity_bytes);
    const auto &unique_id = metadata.uid.value();
    nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
    auto status = nvshmemx_set_attr_uniqueid_args(
        metadata.pe, static_cast<int>(metadata.pe_count()), &unique_id, &attributes);
    TORCH_CHECK(
        status == 0,
        "xpool failed to configure NVSHMEM unique-id attributes: ", status);
    status = nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attributes);
    TORCH_CHECK(status == 0, "xpool failed to initialize the NVSHMEM host library: ", status);
    const auto init_status = nvshmemx_init_status();
    TORCH_CHECK(
        init_status >= NVSHMEM_STATUS_IS_INITIALIZED && init_status < NVSHMEM_STATUS_INVALID,
        "xpool NVSHMEM device initialization is incomplete: ", init_status);
    TORCH_CHECK(
        nvshmem_my_pe() == metadata.pe && nvshmem_n_pes() == static_cast<int>(metadata.pe_count()),
        "xpool NVSHMEM runtime identity differs from Fabric join metadata");

    // Participant-local ownership begins only after NVSHMEM established the
    // exact PE identity. The module registration must precede symmetric arena
    // creation and remain live through resident drain.
    auto module_registration = FabricModuleRegistration::create();
    auto arena = FabricArena::create(
        layout, std::span<const FabricModelLayout>{model_layouts},
        std::span<const FabricLayerLayout>{layer_layouts}, metadata.scheduler_policy);
    // No participant publishes Joined until every PE initialized the same
    // symmetric layout and reached this collective boundary.
    nvshmem_barrier_all();

    cuda_device_ = cuda_device;
    metadata_ = metadata;
    arena_ = std::move(arena);
    module_registration_ = std::move(module_registration);
    phase_ = Phase::Joined;
  } catch (...) {
    phase_ = Phase::Finalized;
    throw;
  }
}

void FabricRuntime::activate() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Joined, "xpool Fabric activate requires a joined process runtime");
  TORCH_CHECK(metadata_.has_value() && cuda_device_.has_value(), "xpool Fabric join state is incomplete");
  TORCH_CHECK(
      static_cast<std::size_t>(metadata_->pe) >= metadata_->atnagent_count,
      "xpool Fabric activate is valid only for an FfnAgent PE");
  TORCH_CHECK(!resident_stream_, "xpool FfnAgent Resident is already active");

  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  // Only FfnAgent PEs own resident Coordinator/Execution kernels. The stream
  // remains live until drain_pending observes terminal kernel completion.
  resident_stream_ = xpool::utils::device::OwnedCudaStream::create();
  launch_ffnagent_kernel(arena_.view(), arena_.layout(), resident_stream_.get());
  arena_.wait_until_resident_ready(resident_stream_,
                                   kResidentStartupTimeout);
}

void FabricRuntime::check_health() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Joined, "xpool Fabric health requires a joined process runtime");
  TORCH_CHECK(metadata_.has_value() && cuda_device_.has_value(), "xpool Fabric join state is incomplete");
  if (static_cast<std::size_t>(metadata_->pe) < metadata_->atnagent_count) {
    return;
  }
  TORCH_CHECK(resident_stream_, "xpool FfnAgent Resident has not been activated");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  if (!resident_stream_.query()) {
    return;
  }
  const auto state = arena_.state();
  TORCH_CHECK(state.failure.publication == 1,
              "xpool FfnAgent Resident completed unexpectedly");
}

FabricArenaView FabricRuntime::arena() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(
      phase_ == Phase::Joined || phase_ == Phase::Draining || phase_ == Phase::Drained,
      "xpool Fabric arena is unavailable outside a joined generation");
  return arena_.view();
}

void FabricRuntime::drain_async() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  if (phase_ == Phase::Draining || phase_ == Phase::Drained) {
    return;
  }
  TORCH_CHECK(phase_ == Phase::Joined, "xpool Fabric drain requires a joined process runtime");
  TORCH_CHECK(cuda_device_.has_value(), "xpool Fabric join state is incomplete");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  // Shutdown publication is asynchronous so the Python control plane can keep
  // reporting progress while all PEs cooperatively retire their residents.
  drain_stream_ = xpool::utils::device::OwnedCudaStream::create();
  arena_.request_shutdown(drain_stream_);
  phase_ = Phase::Draining;
}

bool FabricRuntime::drain_pending() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  if (phase_ == Phase::Drained) {
    return false;
  }
  TORCH_CHECK(phase_ == Phase::Draining, "xpool Fabric drain has not been started");
  TORCH_CHECK(cuda_device_.has_value(), "xpool Fabric join state is incomplete");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  if (!drain_stream_.query() || (resident_stream_ && !resident_stream_.query())) {
    return true;
  }
  // Stream completion is the local proof that shutdown publication completed
  // and every resident kernel owned by this PE has exited.
  drain_stream_.destroy();
  resident_stream_.destroy();
  phase_ = Phase::Drained;
  return false;
}

std::optional<FabricFailure> FabricRuntime::failure() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(
      phase_ == Phase::Joined || phase_ == Phase::Draining || phase_ == Phase::Drained,
      "xpool Fabric failure is unavailable outside a joined generation");
  TORCH_CHECK(cuda_device_.has_value(), "xpool Fabric join state is incomplete");
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  const auto state = arena_.state();
  if (state.failure.publication == 0) {
    return std::nullopt;
  }
  TORCH_CHECK(state.failure.publication == 1, "xpool Fabric failure has an invalid publication value");
  return state.failure;
}

std::optional<FabricTraceSnapshot> FabricRuntime::read_trace() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Drained, "xpool Fabric trace requires a drained process runtime");
  TORCH_CHECK(cuda_device_.has_value(), "xpool Fabric join state is incomplete");
  if (arena_.layout().trace.capacity == 0) {
    return std::nullopt;
  }
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  return arena_.read_trace();
}

void FabricRuntime::shutdown() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(phase_ == Phase::Drained, "xpool Fabric shutdown requires completed local drain");
  TORCH_CHECK(cuda_device_.has_value(), "xpool Fabric join state is incomplete");
  phase_ = Phase::Finalized;
  const auto device_guard = c10::cuda::CUDAGuard{*cuda_device_};
  // Teardown order is contractual: release symmetric allocations while
  // NVSHMEM and its CUDA module remain live, unregister the module, then
  // finalize the participant-local host library.
  arena_.destroy();
  module_registration_.destroy();
  nvshmemx_hostlib_finalize();
  cuda_device_.reset();
  metadata_.reset();
}

} // namespace xpool::fabric
