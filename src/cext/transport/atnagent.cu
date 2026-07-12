#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>

#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/transport.hpp>
#include <xpool/transport/executor.cuh>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/device.cuh>

namespace xpool::transport {

namespace {

__global__ void atnagent_transport_kernel(TransportArena arena) {
  auto used_queue = arena.used_queue();
  constexpr unsigned int kWarpMask = 0xFFFFFFFFU;
  while (true) {
    // Phase 1: warp lane 0 polls shutdown and claims one published slot, then
    // broadcasts the decision to the resident progress warp. The current launch
    // geometry is exactly one warp.
    std::uint32_t slot = 0U;
    unsigned int claimed = 0U;
    unsigned int shutdown_requested = 0U;
    unsigned int should_exit = 0U;
    unsigned int claim_failed = 0U;
    unsigned int abi_mismatch = 0U;
    if (threadIdx.x == 0) {
      shutdown_requested = xpool::atomic::load_acquire(arena.shutdown());
      claimed = used_queue.try_pop(slot) ? 1U : 0U;
      // Shutdown drains every slot already published to the used queue. Once
      // no published slot remains, the resident kernel may exit and the host
      // destroys the arena.
      should_exit = shutdown_requested != 0U && claimed == 0U;
      if (claimed == 0U && should_exit == 0U) {
        __nanosleep(1000U);
      }
    }
    slot = __shfl_sync(kWarpMask, slot, 0);
    claimed = __shfl_sync(kWarpMask, claimed, 0);
    shutdown_requested = __shfl_sync(kWarpMask, shutdown_requested, 0);
    should_exit = __shfl_sync(kWarpMask, should_exit, 0);

    if (claimed == 0U) {
      if (should_exit != 0U) {
        break;
      }
      continue;
    }

    // Phase 2: claim the descriptor. The CAS prevents accidental execution of
    // an un-published or already-claimed slot.
    auto &request = arena.request(slot);
    auto &result = arena.result(slot);
    xpool::abi::TransportTraceRecord *trace_record =
        arena.trace(request.trace_id);
    if (threadIdx.x == 0 && trace_record != nullptr) {
      trace_record->atnagent_dequeued = transport_global_timer();
    }

    if (threadIdx.x == 0) {
      const std::uint32_t previous = xpool::atomic::compare_exchange_acq_rel(
          request.state.status, xpool::abi::DescriptorStatus::kPublished,
          xpool::abi::DescriptorStatus::kGranted);
      claim_failed = previous != xpool::abi::DescriptorStatus::kPublished;
      abi_mismatch = request.header.abi_version != xpool::abi::kAbiVersion;
      if (!claim_failed && !abi_mismatch && trace_record != nullptr) {
        trace_record->descriptor_granted = transport_global_timer();
      }
    }
    claim_failed = __shfl_sync(kWarpMask, claim_failed, 0);
    abi_mismatch = __shfl_sync(kWarpMask, abi_mismatch, 0);

    if (claim_failed != 0U || abi_mismatch != 0U) {
      xpool::utils::device::trap();
    }

    if (threadIdx.x == 0) {
      shutdown_requested = xpool::atomic::load_acquire(arena.shutdown());
    }
    shutdown_requested = __shfl_sync(kWarpMask, shutdown_requested, 0);

    // Phase 3: execute the request. In the current ATN-atnagent loopback
    // checkpoint, the warp writes the output slot directly.
    if (threadIdx.x == 0 && trace_record != nullptr) {
      trace_record->executor_begin = transport_global_timer();
    }
    const std::uint32_t error_code =
        shutdown_requested != 0U ? xpool::abi::FfnResultErrorCode::kShutdown
                                 : execute_transport_request(arena, request);
    __syncwarp(kWarpMask);
    if (threadIdx.x == 0 && trace_record != nullptr) {
      trace_record->executor_end = transport_global_timer();
    }
    // Each warp lane must publish its output writes before lane 0 publishes
    // the result descriptor/status back to the instance process.
    __threadfence_system();
    __syncwarp(kWarpMask);

    // Phase 4: warp lane 0 publishes completion for the slot. The result
    // payload must be written before the release-store status publication.
    if (threadIdx.x == 0) {
      result.header.abi_version = xpool::abi::kAbiVersion;
      result.error_code = error_code;
      result.state.slot_id = request.state.slot_id;
      result.output_offset = request.offsets.output_offset;
      if (error_code != xpool::abi::FfnResultErrorCode::kOk &&
          error_code != xpool::abi::FfnResultErrorCode::kShutdown) {
        (void)xpool::atomic::compare_exchange_acq_rel(
            arena.error_code(), xpool::abi::FfnResultErrorCode::kOk,
            error_code);
      }
      xpool::atomic::store_release(
          result.state.status, error_code == xpool::abi::FfnResultErrorCode::kOk
                                   ? xpool::abi::DescriptorStatus::kDone
                                   : xpool::abi::DescriptorStatus::kFailed);
      if (trace_record != nullptr) {
        trace_record->result_published = transport_global_timer();
      }
    }
    // Re-converge after the lane-0 publish before observing shutdown or
    // polling the next slot.
    __syncwarp(kWarpMask);
  }
}

} // namespace

void launch_atnagent_transport_kernel(TransportArena arena,
                                      cudaStream_t stream) {
  atnagent_transport_kernel<<<1, kAtnAgentThreadsPerBlock, 0, stream>>>(arena);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xpool::transport
