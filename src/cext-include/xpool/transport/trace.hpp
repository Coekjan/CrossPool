#pragma once

/// \file xpool/transport/trace.hpp
/// \brief Semantic trace state machine for one Transport mailbox request.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <xpool/abi.hpp>
#include <xpool/abort.hpp>
#include <xpool/macros.hpp>
#include <xpool/trace.hpp>
#include <xpool/transport/protocol.hpp>
namespace xpool::transport {

/// Ordered event kinds in one Transport mailbox request timeline.
enum class TransportTraceEvent : std::uint32_t {
  /// Instance began preparing the request payload.
  StagingStarted,
  /// Instance completed cooperative request-payload staging.
  StagingCompleted,
  /// Instance recorded the causal marker immediately before request release-store.
  Published,
  /// AtnAgent acquire-observed request publication.
  PublishedObserved,
  /// AtnAgent entered the selected execution path.
  ExecutionStarted,
  /// Production Fabric or selected local AtnAgent loopback admitted output-producing execution.
  ExecutionAdmitted,
  /// The selected execution path finished writing its output.
  ExecutionCompleted,
  /// AtnAgent recorded the causal marker immediately before result release-store.
  Evaluated,
  /// Instance observed the terminal request result.
  EvaluatedObserved,
  /// Instance completed copying result payload to its output tensor.
  OutputCopied,
  /// Instance recorded the causal marker immediately before acknowledgement release-store.
  Acknowledged,
  /// Request entered the terminal closed state during drain.
  Closed,
  /// Number of concrete Transport trace events.
  Count,
};

/// Immutable request facts and dependency-checked timestamps for one request.
class alignas(8) TransportTraceRecord {
public:
  /// Return the arena-local monotonically increasing trace identity.
  /// \return Positive trace sequence assigned when staging starts.
  XPOOL_HOST_DEVICE_FN std::uint64_t trace_id() const { return trace_id_; }
  /// Return the physical hidden-state row count carried by the request.
  /// \return Number of staged payload rows.
  XPOOL_HOST_DEVICE_FN std::size_t payload_rows() const { return payload_rows_; }
  /// Return the model-local FFN layer ordinal.
  /// \return Layer ordinal copied from the request metadata.
  XPOOL_HOST_DEVICE_FN std::size_t layer_ordinal() const { return request_.layer_ordinal; }
  /// Return the raw forward-mode value.
  /// \return ABI value of XPoolForwardMode.
  XPOOL_HOST_DEVICE_FN std::uint32_t forward_mode() const { return request_.forward_mode; }
  /// Return the raw result-handoff value.
  /// \return ABI value of FfnResultHandoff.
  XPOOL_HOST_DEVICE_FN std::uint32_t result_handoff() const { return request_.result_handoff; }
  /// Return the raw DP-padding mode value.
  /// \return ABI value of DpPaddingMode.
  XPOOL_HOST_DEVICE_FN std::uint32_t dp_padding_mode() const { return request_.dp_padding_mode; }
  /// Return the effective request result retained by this trace.
  /// \return Raw ABI value of FfnResultCode.
  XPOOL_HOST_DEVICE_FN std::uint32_t result_code() const { return result_code_.value(); }
  /// Test whether one event timestamp has been recorded.
  /// \param event Event whose presence should be queried.
  /// \return True when the event is present in the timeline.
  XPOOL_HOST_DEVICE_FN bool recorded(TransportTraceEvent event) const { return timeline_.recorded(event); }
  /// Return one raw device timestamp.
  /// \param event Event whose timestamp should be read.
  /// \return Recorded global-timer value, or zero when absent.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(TransportTraceEvent event) const {
    return timeline_.timestamp(event);
  }

#if defined(__CUDACC__)
  /// Initialize a reserved record and mark staging start.
  /// \param trace_id Positive arena-local trace sequence.
  /// \param payload_rows Positive physical payload row count.
  /// \param request Valid request metadata copied into the trace.
  XPOOL_DEVICE_FN void begin(std::uint64_t trace_id, std::size_t payload_rows,
                             const FfnRequestMetadata &request);

  /// Record completion of cooperative payload staging.
  XPOOL_DEVICE_FN void staging_completed();

  /// Record request publication before the mailbox release-store.
  XPOOL_DEVICE_FN void published();

  /// Record the AtnAgent's acquire-observation of publication.
  XPOOL_DEVICE_FN void published_observed();

  /// Record entry into AtnAgent request execution.
  XPOOL_DEVICE_FN void execution_started();

  /// Record admission to production Fabric or the selected local AtnAgent loopback.
  XPOOL_DEVICE_FN void execution_admitted();

  /// Record completion of the selected execution path.
  XPOOL_DEVICE_FN void execution_completed();

  /// Record the AtnAgent result before publishing Evaluated.
  /// \param result_code Request-local or canonical result being published.
  XPOOL_DEVICE_FN void evaluated(xpool::abi::FfnResultCode result_code);

  /// Record the Instance's effective result after canonical-failure precedence.
  /// \param effective_result_code Result linearized by the Instance.
  XPOOL_DEVICE_FN void evaluated_observed(xpool::abi::FfnResultCode effective_result_code);

  /// Record completion of successful output copying.
  XPOOL_DEVICE_FN void output_copied();

  /// Record successful acknowledgement before returning the mailbox to Idle.
  XPOOL_DEVICE_FN void acknowledged();

  /// Record terminal closure of an owned Staging request.
  /// \param result_code Failure or shutdown result retained for diagnostics.
  XPOOL_DEVICE_FN void closed(xpool::abi::FfnResultCode result_code);

  /// Record terminal closure after observing an Evaluated result.
  XPOOL_DEVICE_FN void closed();
#endif

private:
  std::uint64_t trace_id_ = 0;
  std::size_t payload_rows_ = 0;
  FfnRequestMetadata request_{};
  xpool::abi::FfnResultCode result_code_{};
  xpool::trace::Timeline<TransportTraceEvent> timeline_{};
};

/// Host snapshot of the bounded Transport trace buffer after Resident drain.
struct TransportTraceSnapshot {
  std::uint64_t sequence; ///< Total number of trace reservations attempted.
  std::uint64_t dropped;  ///< Reservations omitted after capacity was exhausted.
  std::vector<TransportTraceRecord> records; ///< Retained records in sequence order.
};

static_assert(std::is_trivially_copyable_v<TransportTraceRecord>);

} // namespace xpool::transport
