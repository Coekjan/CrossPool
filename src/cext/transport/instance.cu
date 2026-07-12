#include <ATen/ATen.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>

#include <xpool/transport.hpp>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/device.cuh>

namespace xpool::transport {

namespace {

template <typename scalar_t>
__global__ void
instance_transport_kernel(TransportArena arena, const scalar_t *hidden_input,
                          scalar_t *hidden_output, std::int64_t hidden_numel,
                          const void *dp_token_counts,
                          bool dp_token_counts_are_i64,
                          xpool::abi::FfnRequestMetadata request_metadata,
                          xpool::abi::FfnTensorMetadata tensor_metadata) {
  __shared__ std::uint32_t claimed_slot;
  __shared__ std::uint32_t result_error_code;
  __shared__ bool failed;
  __shared__ xpool::abi::TransportTraceRecord *trace_record;

  if (xpool::atomic::load_acquire(arena.error_code()) !=
      xpool::abi::FfnResultErrorCode::kOk) {
    for (std::int64_t index = threadIdx.x; index < hidden_numel;
         index += blockDim.x) {
      hidden_output[index] = static_cast<scalar_t>(NAN);
    }
    return;
  }

  // Phase 1: reserve one reusable arena slot. Thread 0 owns queue mutation;
  // the block-wide barrier publishes the claimed slot id to staging threads.
  if (threadIdx.x == 0) {
    trace_record = arena.begin_trace(tensor_metadata.num_tokens);
    result_error_code = xpool::abi::FfnResultErrorCode::kOk;
    auto free_queue = arena.free_queue();
    claimed_slot = 0U;
    failed = xpool::atomic::load_acquire(arena.shutdown()) != 0U;
    if (!failed && !free_queue.try_pop_with_timeout(claimed_slot,
                                                    kDeviceSpinTimeoutClocks)) {
      failed = true;
    }
    if (!failed && xpool::atomic::load_acquire(arena.shutdown()) != 0U) {
      (void)free_queue.try_push_with_timeout(claimed_slot,
                                             kDeviceSpinTimeoutClocks);
      failed = true;
    }
    if (!failed && trace_record != nullptr) {
      trace_record->slot = claimed_slot;
      trace_record->slot_claimed = transport_global_timer();
    }
  }
  __syncthreads();
  if (failed) {
    xpool::utils::device::trap();
  }

  // Phase 2: stage the full request payload into arena-owned slot storage.
  auto *staged_input = arena.input_buffer<scalar_t>(claimed_slot);
  for (std::int64_t index = threadIdx.x; index < hidden_numel;
       index += blockDim.x) {
    staged_input[index] = hidden_input[index];
  }

  auto *staged_dp_token_counts = arena.dp_token_counts_buffer(claimed_slot);
  if (dp_token_counts != nullptr) {
    for (std::int64_t index = threadIdx.x;
         index < static_cast<std::int64_t>(request_metadata.atn_dp_size);
         index += blockDim.x) {
      staged_dp_token_counts[index] =
          dp_token_counts_are_i64 ? static_cast<std::uint32_t>(
                                        reinterpret_cast<const std::int64_t *>(
                                            dp_token_counts)[index])
                                  : static_cast<std::uint32_t>(
                                        reinterpret_cast<const std::int32_t *>(
                                            dp_token_counts)[index]);
    }
  }
  // Every staging thread must make its global writes system-visible before
  // thread 0 publishes the descriptor and used-queue slot to the atnagent
  // process.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0 && trace_record != nullptr) {
    trace_record->input_staged = transport_global_timer();
  }

  // Phase 3: publish the descriptor, then enqueue the slot for the atnagent.
  auto &request = arena.request(claimed_slot);
  auto &result = arena.result(claimed_slot);
  auto &request_status = request.state.status;
  auto &result_status = result.state.status;
  if (threadIdx.x == 0) {
    auto used_queue = arena.used_queue();
    if (xpool::atomic::load_acquire(arena.shutdown()) != 0U) {
      failed = true;
    } else {
      request.header.abi_version = xpool::abi::kAbiVersion;
      request.state.slot_id = claimed_slot;
      request.request_metadata = request_metadata;
      request.tensor_metadata = tensor_metadata;
      request.offsets =
          arena.ffn_offsets(claimed_slot, dp_token_counts != nullptr);
      request.trace_id = trace_record == nullptr ? 0 : trace_record->trace_id;
      xpool::atomic::store_release(request_status,
                                   xpool::abi::DescriptorStatus::kPublished);
      failed = !used_queue.try_push_with_timeout(claimed_slot,
                                                 kDeviceSpinTimeoutClocks);
      if (!failed && trace_record != nullptr) {
        trace_record->request_published = transport_global_timer();
      }
    }
  }
  __syncthreads();
  if (failed) {
    if (threadIdx.x == 0) {
      xpool::atomic::store_release(request_status,
                                   xpool::abi::DescriptorStatus::kEmpty);
      xpool::atomic::store_release(result_status,
                                   xpool::abi::DescriptorStatus::kEmpty);
      (void)arena.free_queue().try_push_with_timeout(claimed_slot,
                                                     kDeviceSpinTimeoutClocks);
    }
    __syncthreads();
    xpool::utils::device::trap();
  }

  // Phase 4: wait for the atnagent to publish completion for this slot.
  if (threadIdx.x == 0) {
    const unsigned long long start_clock = clock64();
    while (true) {
      const std::uint32_t status = xpool::atomic::load_acquire(result_status);
      if (status == xpool::abi::DescriptorStatus::kDone ||
          status == xpool::abi::DescriptorStatus::kFailed) {
        failed = result.header.abi_version != xpool::abi::kAbiVersion;
        result_error_code = result.error_code;
        if (!failed &&
            result_error_code == xpool::abi::FfnResultErrorCode::kOk &&
            trace_record != nullptr) {
          trace_record->result_observed = transport_global_timer();
        }
        break;
      }
      if (clock64() - start_clock > kDeviceSpinTimeoutClocks) {
        failed = true;
        break;
      }
      __nanosleep(1000U);
    }
  }
  __syncthreads();
  if (!failed && result_error_code != xpool::abi::FfnResultErrorCode::kOk) {
    for (std::int64_t index = threadIdx.x; index < hidden_numel;
         index += blockDim.x) {
      hidden_output[index] = static_cast<scalar_t>(NAN);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      xpool::atomic::store_release(request_status,
                                   xpool::abi::DescriptorStatus::kEmpty);
      xpool::atomic::store_release(result_status,
                                   xpool::abi::DescriptorStatus::kEmpty);
      failed = !arena.free_queue().try_push_with_timeout(
          claimed_slot, kDeviceSpinTimeoutClocks);
    }
    __syncthreads();
    if (failed) {
      xpool::utils::device::trap();
    }
    return;
  }
  if (failed) {
    if (threadIdx.x == 0) {
      xpool::atomic::store_release(request_status,
                                   xpool::abi::DescriptorStatus::kEmpty);
      xpool::atomic::store_release(result_status,
                                   xpool::abi::DescriptorStatus::kEmpty);
      (void)arena.free_queue().try_push_with_timeout(claimed_slot,
                                                     kDeviceSpinTimeoutClocks);
    }
    __syncthreads();
    xpool::utils::device::trap();
  }

  // Phase 5: copy the completed output back to the PyTorch return tensor.
  const auto *staged_output = arena.output_buffer<scalar_t>(claimed_slot);
  for (std::int64_t index = threadIdx.x; index < hidden_numel;
       index += blockDim.x) {
    hidden_output[index] = staged_output[index];
  }
  // Do not recycle the slot until every output element has been copied out.
  __syncthreads();
  if (threadIdx.x == 0 && trace_record != nullptr) {
    trace_record->output_copied = transport_global_timer();
  }

  // Phase 6: clear descriptor state and return the slot to the free queue.
  if (threadIdx.x == 0) {
    auto free_queue = arena.free_queue();
    xpool::atomic::store_release(request_status,
                                 xpool::abi::DescriptorStatus::kEmpty);
    xpool::atomic::store_release(result_status,
                                 xpool::abi::DescriptorStatus::kEmpty);
    failed = !free_queue.try_push_with_timeout(claimed_slot,
                                               kDeviceSpinTimeoutClocks);
    if (!failed && trace_record != nullptr) {
      __threadfence_system();
      trace_record->slot_recycled = transport_global_timer();
    }
  }
  __syncthreads();
  if (failed) {
    xpool::utils::device::trap();
  }
}

} // namespace

void launch_instance_transport_kernel(const TransportRequest &request) {
  const void *dp_token_counts = nullptr;
  c10::ScalarType dp_token_counts_scalar_type = at::kInt;
  if (request.global_num_tokens_gpu.has_value() &&
      request.global_num_tokens_gpu->defined() &&
      request.global_num_tokens_gpu->numel() != 0) {
    const at::Tensor &token_counts = *request.global_num_tokens_gpu;
    dp_token_counts = token_counts.const_data_ptr();
    dp_token_counts_scalar_type = token_counts.scalar_type();
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, request.hidden_states.scalar_type(),
      "xpool_transport_ffn_shim", [&] {
        instance_transport_kernel<scalar_t>
            <<<1, kRequestThreadsPerBlock, 0, request.stream>>>(
                request.arena, request.hidden_states.const_data_ptr<scalar_t>(),
                request.output.data_ptr<scalar_t>(),
                request.hidden_states.numel(), dp_token_counts,
                dp_token_counts_scalar_type == at::kLong,
                request.request_metadata, request.tensor_metadata);
      });
}

} // namespace xpool::transport
