#pragma once

/// \file xpool/transport/protocol.hpp
/// \brief Request metadata and single-producer/single-consumer mailbox protocol.

#include <cstddef>
#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/macros.hpp>
#include <xpool/transport/layout.hpp>

namespace xpool::transport {

/// Request-varying FFN semantics published through one Transport mailbox.
struct FfnRequestMetadata {
  /// Zero-based ordinal in the model's canonical FFN layer sequence.
  std::size_t layer_ordinal;
  /// Stable xpool::abi::XPoolForwardMode value.
  std::uint32_t forward_mode;
  /// Stable xpool::abi::FfnResultHandoff value.
  std::uint32_t result_handoff;
  /// Stable xpool::abi::DpPaddingMode value.
  std::uint32_t dp_padding_mode;

  /// Return whether every enum-valued field belongs to its declared domain.
  /// \return True when all request enum values are valid.
  XPOOL_HOST_DEVICE_FN bool valid() const {
    return xpool::abi::XPoolForwardMode::is_valid(forward_mode) &&
           xpool::abi::FfnResultHandoff::is_valid(result_handoff) &&
           xpool::abi::DpPaddingMode::is_valid(dp_padding_mode);
  }

  /// Validate this request against one immutable Transport arena contract.
  /// \param layout Arena geometry and topology for the attached Instance.
  /// \param payload_rows Physical hidden-state rows carried by this request.
  /// \param token_counts_present Whether the DP token-count vector is present.
  void validate(const TransportArenaLayout &layout, std::size_t payload_rows,
                bool token_counts_present) const;
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

/// One rank-local FFN request mailbox shared by an Instance and AtnAgent.
struct alignas(8) TransportMailbox {
  /// System-scope atomic MailboxStatus value.
  std::uint32_t status;
  /// Stable xpool::abi::FfnResultCode published with Evaluated.
  std::uint32_t result_code;
  /// Physical hidden-state rows in the current request.
  std::size_t payload_rows;
  /// Request semantics retained until acknowledgement or terminal close.
  FfnRequestMetadata request;

#if defined(__CUDACC__)
  /// Acquire-observe the current validated lifecycle state.
  /// \return Current MailboxStatus.
  XPOOL_DEVICE_FN MailboxStatus observe();
  /// Release-publish the sole Dormant-to-Idle Resident readiness edge.
  /// \pre The mailbox is Dormant and the complete Resident grid passed its
  /// startup barrier.
  XPOOL_DEVICE_FN void open();
  /// Race the terminal consumer for ownership of an Idle mailbox.
  /// \return True when this Instance acquired Staging ownership.
  XPOOL_DEVICE_FN bool try_begin_staging();
  /// Release-publish a complete staged request to the AtnAgent.
  XPOOL_DEVICE_FN void publish_request();
  /// Release-publish one request-local result to the Instance.
  /// \param result_code Valid result for the currently published request.
  XPOOL_DEVICE_FN void publish_result(xpool::abi::FfnResultCode result_code);
  /// Reset request facts and release the successfully consumed mailbox to Idle.
  XPOOL_DEVICE_FN void acknowledge();
  /// Terminally close an Instance-owned Staging request.
  XPOOL_DEVICE_FN void close_staging();
  /// Terminally close an Instance-owned Evaluated request.
  XPOOL_DEVICE_FN void close_evaluated();
  /// Race the producer for terminal ownership of an Idle mailbox.
  /// \return True when the AtnAgent changed Idle to Closed.
  XPOOL_DEVICE_FN bool try_close_idle();
#endif
};

} // namespace xpool::transport
