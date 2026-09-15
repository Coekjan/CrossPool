#pragma once

/// \file xpool/transport/protocol.hpp
/// \brief Request metadata and single-producer/single-consumer mailbox protocol.

#include <cstddef>
#include <cstdint>

#include <xpool/ffn.hpp>
#include <xpool/macros.hpp>

namespace xpool::transport {

/// Request-varying FFN semantics published through one Transport mailbox.
struct RequestMetadata {
  /// Zero-based ordinal in the model's canonical FFN layer sequence.
  std::size_t layer_ordinal;
  /// Runtime forward mode.
  xpool::ffn::ForwardMode forward_mode;
  /// Mathematical output requirement.
  xpool::ffn::OutputRequirement output_requirement;
  /// Physical DP row layout.
  xpool::ffn::DpRowLayout dp_row_layout;

  /// Return whether every enum-valued field belongs to its declared domain.
  XPOOL_HOST_DEVICE_FN bool valid() const {
    return xpool::ffn::is_valid(forward_mode) && xpool::ffn::is_valid(output_requirement) &&
           xpool::ffn::is_valid(dp_row_layout);
  }
};

/// Observable lifecycle of one reusable single-producer/single-consumer mailbox.
enum class MailboxStatus : std::uint32_t {
  /// Zero-initialized endpoint whose Resident block has not become ready.
  Dormant = 0,
  /// Ready endpoint with no outstanding request.
  Idle = 1,
  /// Instance-owned request staging is in progress.
  Staging = 2,
  /// AtnAgent-owned request evaluation is available.
  Published = 3,
  /// Instance-owned result consumption is available.
  Evaluated = 4,
  /// Terminal endpoint that cannot be reused.
  Closed = 5,
};

/// One rank-local FFN request mailbox shared by an Instance rank and AtnAgent.
struct alignas(8) Mailbox {
  /// System-scope atomic MailboxStatus value.
  MailboxStatus status;
  /// Result published with Evaluated.
  xpool::ffn::ResultCode result_code;
  /// Physical hidden-state rows in the current request.
  std::size_t payload_rows;
  /// Request semantics retained until acknowledgement or terminal close.
  RequestMetadata request;

#if defined(__CUDACC__)
  /// Acquire-observe the current validated lifecycle state.
  XPOOL_DEVICE_FN MailboxStatus observe();
  /// Release-publish the sole Dormant-to-Idle Resident readiness edge.
  /// \pre The mailbox is Dormant and the complete Resident grid passed its
  /// startup barrier.
  XPOOL_DEVICE_FN void open();
  /// Race the terminal consumer for ownership of an Idle mailbox.
  XPOOL_DEVICE_FN bool try_begin_staging();
  /// Release-publish a complete staged request to the AtnAgent.
  /// \pre The caller owns Staging and has completed payload and metadata writes.
  XPOOL_DEVICE_FN void publish_request();
  /// Release-publish one request-local result to the Instance rank.
  /// \pre The AtnAgent owns Published.
  XPOOL_DEVICE_FN void publish_result(xpool::ffn::ResultCode result_code);
  /// Reset request facts and release the successfully consumed mailbox to Idle.
  /// \pre The Instance rank owns an Ok Evaluated result. Request facts are
  /// cleared before Idle becomes visible to the next producer.
  XPOOL_DEVICE_FN void acknowledge();
  /// Terminally close an Instance-owned Staging request.
  /// \pre The Instance rank owns Staging.
  XPOOL_DEVICE_FN void close_staging();
  /// Terminally close an Instance-owned Evaluated request.
  /// \pre The Instance rank owns Evaluated.
  XPOOL_DEVICE_FN void close_evaluated();
  /// Race the producer for terminal ownership of an Idle mailbox.
  XPOOL_DEVICE_FN bool try_close_idle();
#endif
};

} // namespace xpool::transport
