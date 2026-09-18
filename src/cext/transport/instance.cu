#include <xpool/transport/instance.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <ATen/ATen.h>
#include <c10/util/TypeCast.h>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include <xpool/abort.hpp>
#include <xpool/hooks.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/cooperative.cuh>
#include <xpool/utils/wait.cuh>

namespace xpool::transport {

namespace {

constexpr int kRequestThreadsPerBlock = 256;

template <typename Scalar>
XPOOL_DEVICE_FN void fill_failure_output(const cooperative_groups::thread_block &group, Scalar *output,
                                         std::size_t element_count) {
  xpool::utils::cooperative::fill(group, cuda::std::span{output, element_count}, static_cast<Scalar>(NAN));
}

// Execute one synchronous Instance-side mailbox transaction on the device;
// graph capture preserves its addresses and control flow.
template <typename Scalar>
XPOOL_KERNEL_FN void instance_transport_kernel(ArenaView arena, const Scalar *hidden_input, Scalar *hidden_output,
                                               std::size_t hidden_numel, std::size_t payload_rows,
                                               const void *dp_rank_payload_rows, bool dp_rank_payload_rows_are_i64,
                                               xpool::transport::RequestMetadata request_metadata) {
  XPOOL_DEVICE_SHARED xpool::utils::wait::Status wait_result;
  XPOOL_DEVICE_SHARED xpool::ffn::ResultCode result_code;
  XPOOL_DEVICE_SHARED MailboxStatus observed_status;
  XPOOL_DEVICE_SHARED bool terminal_before_staging;
  XPOOL_DEVICE_SHARED bool canonical_failure_after_timeout;

  const auto group = cooperative_groups::this_thread_block();
  auto &mailbox = arena.mailbox();

  // Phase: Claim - Thread zero exclusively moves an available mailbox into Staging
  // and reserves the request's trace identity.
  if (group.thread_rank() == 0) {
    const auto failure = arena.generation_failure();
    const auto shutdown = arena.shutdown_requested();
    terminal_before_staging = failure != xpool::ffn::ResultCode::Ok || shutdown;
    result_code = failure != xpool::ffn::ResultCode::Ok ? failure : xpool::ffn::ResultCode::Shutdown;
    if (!terminal_before_staging) {
      if (!mailbox.try_begin_staging()) {
        xpool::abort_if(mailbox.observe() != MailboxStatus::Closed);
        const auto terminal_failure = arena.generation_failure();
        const auto terminal_shutdown = arena.shutdown_requested();
        xpool::abort_if(terminal_failure == xpool::ffn::ResultCode::Ok && !terminal_shutdown);
        terminal_before_staging = true;
        result_code =
            terminal_failure != xpool::ffn::ResultCode::Ok ? terminal_failure : xpool::ffn::ResultCode::Shutdown;
      } else {
        xpool::hooks::TransportInstanceProtocolEvent::hooks(
            {.arena = arena,
             .kind = xpool::hooks::TransportProtocolEventKind::RequestStagingStarted,
             .payload_rows = payload_rows,
             .request = &request_metadata});
      }
    }
  }
  group.sync();
  if (terminal_before_staging) {
    fill_failure_output(group, hidden_output, hidden_numel);
    return;
  }

  // Phase: Stage - The block copies all request-owned payload and DP metadata before
  // any part of the request becomes visible to the Resident.
  const auto payload_bytes = hidden_numel * sizeof(Scalar);
  xpool::utils::cooperative::copy(group, arena.input_payload().first(payload_bytes),
                                  cuda::std::span{reinterpret_cast<const std::uint8_t *>(hidden_input), payload_bytes});
  auto staged_payload_rows = arena.dp_rank_payload_rows();
  if (dp_rank_payload_rows != nullptr) {
    const auto convert = [] XPOOL_DEVICE_FN(auto row_count) {
      xpool::abort_if(row_count < 0 ||
                      static_cast<std::uint64_t>(row_count) > std::numeric_limits<std::uint32_t>::max());
      return static_cast<std::uint32_t>(row_count);
    };
    if (dp_rank_payload_rows_are_i64) {
      xpool::utils::cooperative::transform(
          group, staged_payload_rows,
          cuda::std::span{static_cast<const std::int64_t *>(dp_rank_payload_rows), staged_payload_rows.size()},
          convert);
    } else {
      xpool::utils::cooperative::transform(
          group, staged_payload_rows,
          cuda::std::span{static_cast<const std::int32_t *>(dp_rank_payload_rows), staged_payload_rows.size()},
          convert);
    }
  } else {
    group.sync();
  }

  // Phase: Publish - Thread zero commits metadata and performs the sole Staging to
  // Published transition after the staged payload is globally complete.
  if (group.thread_rank() == 0) {
    mailbox.payload_rows = payload_rows;
    mailbox.request = request_metadata;
    xpool::hooks::TransportInstanceProtocolEvent::hooks(
        {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::RequestStagingCompleted});
    const auto failure = arena.generation_failure();
    const auto shutdown = arena.shutdown_requested();
    if (failure != xpool::ffn::ResultCode::Ok || shutdown) {
      result_code = failure != xpool::ffn::ResultCode::Ok ? failure : xpool::ffn::ResultCode::Shutdown;
      mailbox.result_code = result_code;
      mailbox.close_staging();
      xpool::hooks::TransportInstanceProtocolEvent::hooks(
          {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::Closed, .result_code = result_code});
      terminal_before_staging = true;
    } else {
      mailbox.publish_request();
      xpool::hooks::TransportInstanceProtocolEvent::hooks(
          {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::RequestPublished});
    }
  }
  group.sync();
  if (terminal_before_staging) {
    fill_failure_output(group, hidden_output, hidden_numel);
    return;
  }

  // Phase: Evaluate - Wait for the Resident's terminal result while canonical Fabric
  // failure retains precedence over a concurrent timeout.
  if (group.thread_rank() == 0) {
    canonical_failure_after_timeout = false;
    const auto deadline = xpool::utils::wait::Deadline::after(xpool::utils::wait::kDefaultTimeoutNanoseconds);
    wait_result = xpool::utils::wait::until(deadline, [&] {
      observed_status = mailbox.observe();
      return observed_status == MailboxStatus::Evaluated || observed_status == MailboxStatus::Closed;
    });
    if (wait_result == xpool::utils::wait::Status::TimedOut) {
      const auto failure = arena.generation_failure();
      if (failure != xpool::ffn::ResultCode::Ok) {
        result_code = failure;
        canonical_failure_after_timeout = true;
      }
    }
  }
  group.sync();
  if (wait_result == xpool::utils::wait::Status::TimedOut) {
    if (!canonical_failure_after_timeout) {
      xpool::abort();
    }
    fill_failure_output(group, hidden_output, hidden_numel);
    return;
  }
  xpool::abort_if(observed_status != MailboxStatus::Evaluated);

  if (group.thread_rank() == 0) {
    const auto failure = arena.generation_failure();
    const auto mailbox_result = mailbox.result_code;
    result_code = mailbox_result == xpool::ffn::ResultCode::Ok || failure == xpool::ffn::ResultCode::Ok ? mailbox_result
                                                                                                        : failure;
    xpool::abort_if(!xpool::ffn::is_valid(result_code));
    xpool::hooks::TransportInstanceProtocolEvent::hooks(
        {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::ResultObserved, .result_code = result_code});
  }
  group.sync();
  if (result_code != xpool::ffn::ResultCode::Ok) {
    fill_failure_output(group, hidden_output, hidden_numel);
    if (group.thread_rank() == 0) {
      mailbox.close_evaluated();
      xpool::hooks::TransportInstanceProtocolEvent::hooks(
          {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::Closed, .result_code = result_code});
    }
    return;
  }

  // Phase: Acknowledge - Successful output is copied before the mailbox returns to
  // Idle; shutdown instead closes the evaluated mailbox.
  xpool::utils::cooperative::copy(group,
                                  cuda::std::span{reinterpret_cast<std::uint8_t *>(hidden_output), payload_bytes},
                                  arena.output_payload().first(payload_bytes));
  if (group.thread_rank() == 0) {
    xpool::hooks::TransportInstanceProtocolEvent::hooks(
        {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::OutputCopied});
    if (arena.shutdown_requested()) {
      mailbox.close_evaluated();
      xpool::hooks::TransportInstanceProtocolEvent::hooks({.arena = arena,
                                                           .kind = xpool::hooks::TransportProtocolEventKind::Closed,
                                                           .result_code = xpool::ffn::ResultCode::Shutdown});
    } else {
      mailbox.acknowledge();
      xpool::hooks::TransportInstanceProtocolEvent::hooks(
          {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::ResultAcknowledged});
    }
  }
}

} // namespace

void launch_request_kernel(const Request &request) {
  auto dp_rank_payload_rows = static_cast<const void *>(nullptr);
  auto dp_rank_payload_rows_scalar_type = at::kInt;
  if (request.dp_rank_payload_rows.has_value() && request.dp_rank_payload_rows->defined() &&
      request.dp_rank_payload_rows->numel() != 0) {
    const auto &rank_payload_rows = *request.dp_rank_payload_rows;
    dp_rank_payload_rows = rank_payload_rows.const_data_ptr();
    dp_rank_payload_rows_scalar_type = rank_payload_rows.scalar_type();
  }
  const auto payload_rows = c10::checked_convert<std::size_t>(request.hidden_states.size(0), "hidden-state rows");
  const auto hidden_numel = c10::checked_convert<std::size_t>(request.hidden_states.numel(), "hidden-state elements");
  const auto launch = [&]<typename Scalar>() {
    instance_transport_kernel<Scalar><<<1, kRequestThreadsPerBlock, 0, request.stream>>>(
        request.arena, request.hidden_states.const_data_ptr<Scalar>(), request.output.data_ptr<Scalar>(), hidden_numel,
        payload_rows, dp_rank_payload_rows, dp_rank_payload_rows_scalar_type == at::kLong, request.request_metadata);
  };
  switch (request.hidden_states.scalar_type()) {
  case c10::ScalarType::Half:
    launch.template operator()<c10::Half>();
    break;
  case c10::ScalarType::BFloat16:
    launch.template operator()<c10::BFloat16>();
    break;
  default:
    TORCH_CHECK(false, "xpool FFN shim requires BF16 or FP16 hidden states");
  }
}

} // namespace xpool::transport
