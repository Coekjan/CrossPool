#include <ATen/ATen.h>
#include <c10/util/TypeCast.h>

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <xpool/abort.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/instance.hpp>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/cooperative.cuh>
#include <xpool/utils/wait.cuh>

namespace xpool::transport {

namespace {

constexpr int kRequestThreadsPerBlock = 256;

template <typename Scalar>
XPOOL_DEVICE_FN void fill_failure(const cooperative_groups::thread_block &group, Scalar *output,
                                  std::size_t element_count) {
  for (auto index = std::size_t{threadIdx.x}; index < element_count; index += blockDim.x) {
    output[index] = static_cast<Scalar>(NAN);
  }
  group.sync();
}

/// Execute one synchronous Instance-side mailbox transaction without host
/// progress, preserving graph-capture-compatible addresses and control flow.
template <typename Scalar>
__global__ void instance_transport_kernel(TransportArenaView arena, const Scalar *hidden_input,
                                          Scalar *hidden_output, std::size_t hidden_numel,
                                          std::size_t payload_rows, const void *dp_token_counts,
                                          bool dp_token_counts_are_i64,
                                          xpool::transport::FfnRequestMetadata request_metadata) {
  __shared__ xpool::utils::wait::Result wait_result;
  __shared__ std::uint32_t result_code;
  __shared__ std::uint32_t observed_status;
  __shared__ TransportTraceRecord *trace_record;
  __shared__ bool terminal_before_staging;
  __shared__ bool canonical_failure_after_timeout;

  const auto group = cooperative_groups::this_thread_block();
  auto &mailbox = arena.mailbox();

  // Phase: Claim - Thread zero exclusively moves an available mailbox into Staging
  // and reserves the request's trace identity.
  if (threadIdx.x == 0) {
    const auto failure = arena.generation_failure();
    const auto shutdown = arena.shutdown_requested();
    terminal_before_staging = failure != xpool::abi::FfnResultCode::Ok || shutdown;
    result_code = failure != xpool::abi::FfnResultCode::Ok
                      ? failure.value()
                      : static_cast<std::uint32_t>(xpool::abi::FfnResultCode::Shutdown);
    trace_record = nullptr;
    if (!terminal_before_staging) {
      if (!mailbox.try_begin_staging()) {
        xpool::abort_if(mailbox.observe() != MailboxStatus::Closed);
        const auto terminal_failure = arena.generation_failure();
        const auto terminal_shutdown = arena.shutdown_requested();
        xpool::abort_if(terminal_failure == xpool::abi::FfnResultCode::Ok && !terminal_shutdown);
        terminal_before_staging = true;
        result_code = terminal_failure != xpool::abi::FfnResultCode::Ok
                          ? terminal_failure.value()
                          : static_cast<std::uint32_t>(xpool::abi::FfnResultCode::Shutdown);
      } else {
        const auto entry = arena.reserve_trace();
        if (entry) {
          trace_record = &entry.record();
          trace_record->begin(entry.sequence(), payload_rows, request_metadata);
        }
      }
    }
  }
  group.sync();
  if (terminal_before_staging) {
    fill_failure(group, hidden_output, hidden_numel);
    return;
  }

  // Phase: Stage - The block copies all request-owned payload and DP metadata before
  // any part of the request becomes visible to the Resident.
  xpool::utils::cooperative::copy(group, arena.input_payload(), hidden_input, hidden_numel * sizeof(Scalar));
  auto *staged_counts = arena.dp_token_counts();
  if (dp_token_counts != nullptr) {
    xpool::abort_if(staged_counts == nullptr);
    for (auto index = std::size_t{threadIdx.x}; index < arena.layout().atn_dp_size;
         index += blockDim.x) {
      const auto count = dp_token_counts_are_i64
                             ? static_cast<const std::int64_t *>(dp_token_counts)[index]
                             : static_cast<std::int64_t>(static_cast<const std::int32_t *>(dp_token_counts)[index]);
      xpool::abort_if(count < 0 ||
                      static_cast<std::uint64_t>(count) > std::numeric_limits<std::uint32_t>::max());
      staged_counts[index] = static_cast<std::uint32_t>(count);
    }
  }
  group.sync();

  // Phase: Publish - Thread zero commits metadata and performs the sole Staging to
  // Published transition after the staged payload is globally complete.
  if (threadIdx.x == 0) {
    if (trace_record != nullptr) {
      trace_record->staging_completed();
    }
    mailbox.payload_rows = payload_rows;
    mailbox.request = request_metadata;
    const auto failure = arena.generation_failure();
    const auto shutdown = arena.shutdown_requested();
    if (failure != xpool::abi::FfnResultCode::Ok || shutdown) {
      result_code = failure != xpool::abi::FfnResultCode::Ok
                        ? failure.value()
                        : static_cast<std::uint32_t>(xpool::abi::FfnResultCode::Shutdown);
      mailbox.result_code = result_code;
      if (trace_record != nullptr) {
        trace_record->closed(xpool::abi::FfnResultCode{result_code});
      }
      mailbox.close_staging();
      terminal_before_staging = true;
    } else {
      if (trace_record != nullptr) {
        trace_record->published();
      }
      mailbox.publish_request();
    }
  }
  group.sync();
  if (terminal_before_staging) {
    fill_failure(group, hidden_output, hidden_numel);
    return;
  }

  // Phase: Evaluate - Wait for the Resident's terminal result while canonical Fabric
  // failure retains precedence over a concurrent timeout.
  if (threadIdx.x == 0) {
    canonical_failure_after_timeout = false;
    const auto deadline = xpool::utils::wait::Deadline::after(xpool::utils::wait::kDefaultTimeoutNanoseconds);
    wait_result = xpool::utils::wait::until(deadline, [&] {
      observed_status = static_cast<std::uint32_t>(mailbox.observe());
      return observed_status == static_cast<std::uint32_t>(MailboxStatus::Evaluated) ||
             observed_status == static_cast<std::uint32_t>(MailboxStatus::Closed);
    });
    if (wait_result == xpool::utils::wait::Result::TimedOut) {
      const auto failure = arena.generation_failure();
      if (failure != xpool::abi::FfnResultCode::Ok) {
        result_code = failure.value();
        canonical_failure_after_timeout = true;
      }
    }
  }
  group.sync();
  if (wait_result == xpool::utils::wait::Result::TimedOut) {
    if (!canonical_failure_after_timeout) {
      xpool::abort();
    }
    fill_failure(group, hidden_output, hidden_numel);
    return;
  }
  xpool::abort_if(observed_status != static_cast<std::uint32_t>(MailboxStatus::Evaluated));

  if (threadIdx.x == 0) {
    const auto failure = arena.generation_failure();
    result_code = failure == xpool::abi::FfnResultCode::Ok ? mailbox.result_code : failure.value();
    xpool::abort_if(!xpool::abi::FfnResultCode::is_valid(result_code));
    if (trace_record != nullptr) {
      trace_record->evaluated_observed(xpool::abi::FfnResultCode{result_code});
    }
  }
  group.sync();
  if (result_code != xpool::abi::FfnResultCode::Ok) {
    fill_failure(group, hidden_output, hidden_numel);
    if (threadIdx.x == 0) {
      if (trace_record != nullptr) {
        trace_record->closed();
      }
      mailbox.close_evaluated();
    }
    return;
  }

  // Phase: Acknowledge - Successful output is copied before the mailbox returns to
  // Idle; shutdown instead closes the evaluated mailbox.
  xpool::utils::cooperative::copy(group, hidden_output, arena.output_payload(), hidden_numel * sizeof(Scalar));
  if (threadIdx.x == 0) {
    if (trace_record != nullptr) {
      trace_record->output_copied();
    }
    if (arena.shutdown_requested()) {
      if (trace_record != nullptr) {
        trace_record->closed();
      }
      mailbox.close_evaluated();
    } else {
      if (trace_record != nullptr) {
        trace_record->acknowledged();
      }
      mailbox.acknowledge();
    }
  }
}

} // namespace

void launch_request_kernel(const TransportRequest &request) {
  auto dp_token_counts = static_cast<const void *>(nullptr);
  auto dp_token_counts_scalar_type = at::kInt;
  if (request.global_num_tokens_gpu.has_value() && request.global_num_tokens_gpu->defined() &&
      request.global_num_tokens_gpu->numel() != 0) {
    const auto &token_counts = *request.global_num_tokens_gpu;
    dp_token_counts = token_counts.const_data_ptr();
    dp_token_counts_scalar_type = token_counts.scalar_type();
  }
  const auto payload_rows = c10::checked_convert<std::size_t>(request.hidden_states.size(0), "hidden-state rows");
  const auto hidden_numel = c10::checked_convert<std::size_t>(request.hidden_states.numel(), "hidden-state elements");
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, request.hidden_states.scalar_type(), "xpool_transport_ffn_shim", [&] {
        instance_transport_kernel<scalar_t><<<1, kRequestThreadsPerBlock, 0, request.stream>>>(
            request.arena, request.hidden_states.const_data_ptr<scalar_t>(), request.output.data_ptr<scalar_t>(),
            hidden_numel, payload_rows, dp_token_counts, dp_token_counts_scalar_type == at::kLong,
            request.request_metadata);
      });
}

} // namespace xpool::transport
