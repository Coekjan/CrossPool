#pragma once

/// \file xpool/fabric/protocol.hpp
/// \brief Typed NVSHMEM records for one distributed FFN invocation.

#include <cuda/std/span>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/fabric/layout.hpp>
#include <xpool/ffn.hpp>
#include <xpool/macros.hpp>

#if defined(__CUDACC__)
#include <cooperative_groups.h>
#endif

namespace xpool::fabric {

/// Delivery branch selected for one distributed FFN invocation.
enum class DeliveryVariant : std::uint32_t {
  /// Each FfnAgent writes its local TP partial directly to one AtnAgent.
  DirectPartial = 1,
  /// One FfnAgent reduces TP partials and writes one complete output.
  SingleComplete = 2,
  /// Every FfnAgent receives the complete reduced output for rank-local delivery.
  ReplicatedComplete = 3,
};

/// Return whether a raw delivery variant names a supported protocol branch.
XPOOL_HOST_DEVICE_FN constexpr bool is_valid(DeliveryVariant value) {
  return value == DeliveryVariant::DirectPartial || value == DeliveryVariant::SingleComplete ||
         value == DeliveryVariant::ReplicatedComplete;
}

/// Identity shared by every record belonging to one FFN invocation.
struct InvocationKey {
  /// Config-order Instance identity in the generation Projection.
  std::size_t instance_index;
  /// Positive sequence assigned monotonically by the Instance rank.
  std::uint64_t invocation_sequence;

  /// Return whether this key can identify a published invocation.
  XPOOL_HOST_DEVICE_FN constexpr bool valid() const { return invocation_sequence != 0; }
  constexpr bool operator==(const InvocationKey &) const = default;
};

/// Coordinator-private aggregate after successful Submission rendezvous.
struct Invocation {
  /// Invocation identity established by Submission rendezvous.
  InvocationKey key;
  /// Config-order FFN layer ordinal within the selected Instance.
  std::size_t layer_ordinal;
  /// Number of live hidden-state rows in the complete invocation.
  std::size_t payload_rows;
  /// Mathematical output requirement requested by the Instance.
  xpool::ffn::OutputRequirement output_requirement;

  /// Validate intrinsic identity, dimensions, and closed-set values.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const {
    return key.valid() && payload_rows != 0 && xpool::ffn::is_valid(output_requirement)
               ? xpool::ffn::ResultCode::Ok
               : xpool::ffn::ResultCode::ProtocolMismatch;
  }
  constexpr bool operator==(const Invocation &) const = default;
};

/// AtnAgent publication contributing one rank-local input description.
struct Submission {
  /// Shared invocation identity.
  InvocationKey key;
  /// Config-order FFN layer ordinal.
  std::size_t layer_ordinal;
  /// Total live rows across participating DP ranks.
  std::size_t payload_rows;
  /// Physical DP row layout describing the submitted payload rows.
  xpool::ffn::DpRowLayout dp_row_layout;
  /// Mathematical output requirement requested by the Instance.
  xpool::ffn::OutputRequirement output_requirement;
  /// Rows contributed by this AtnAgent's DP rank.
  std::size_t dp_rank_payload_rows;
  /// Runtime forward mode selecting Decode or Prefill capacity.
  xpool::ffn::ForwardMode forward_mode;

  /// Return the sequence that publishes this record.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return key.invocation_sequence; }
  /// Validate intrinsic submission identity, row geometry, and closed-set values.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const {
    return key.valid() && payload_rows != 0 && dp_rank_payload_rows <= payload_rows &&
                   xpool::ffn::is_valid(forward_mode) && xpool::ffn::is_valid(output_requirement) &&
                   xpool::ffn::is_valid(dp_row_layout)
               ? xpool::ffn::ResultCode::Ok
               : xpool::ffn::ResultCode::ProtocolMismatch;
  }
  constexpr bool operator==(const Submission &) const = default;
};

/// Coordinator publication assigning one invocation to an Executor Lane lease.
struct Admission {
  /// Admitted invocation identity.
  InvocationKey key;
  /// Generation-local Executor Lane index.
  std::size_t executor_lane_index;
  /// Positive lease sequence preventing stale Lane reuse.
  std::uint64_t executor_lease_sequence;

  /// Return the sequence that publishes this record.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return key.invocation_sequence; }
  /// Validate admission identity and lease sequence.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const {
    return key.valid() && executor_lease_sequence != 0 ? xpool::ffn::ResultCode::Ok
                                                       : xpool::ffn::ResultCode::ProtocolMismatch;
  }
  constexpr bool operator==(const Admission &) const = default;
};

/// Coordinator publication launching one admitted invocation on its Lane.
struct LaneExecution {
  /// Admitted invocation identity.
  InvocationKey key;
  /// Positive sequence of the active Executor Lane lease.
  std::uint64_t executor_lease_sequence;
  /// Config-order FFN layer ordinal to execute.
  std::size_t layer_ordinal;
  /// Number of live hidden-state rows.
  std::size_t payload_rows;
  /// Mathematical output requirement controlling delivery.
  xpool::ffn::OutputRequirement output_requirement;

  /// Return the sequence that publishes this Lane-scoped record.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return executor_lease_sequence; }
  /// Validate execution identity, lease, dimensions, and requirement.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const {
    return key.valid() && executor_lease_sequence != 0 && payload_rows != 0 && xpool::ffn::is_valid(output_requirement)
               ? xpool::ffn::ResultCode::Ok
               : xpool::ffn::ResultCode::ProtocolMismatch;
  }
  constexpr bool operator==(const LaneExecution &) const = default;
};

/// Validate the identity shared by successful Lane-scoped records.
/// \tparam Record Lane-scoped record containing `key` and `executor_lease_sequence`.
template <class Record>
XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate_lane_success(const Record &record) {
  return record.key.valid() && record.executor_lease_sequence != 0 ? xpool::ffn::ResultCode::Ok
                                                                   : xpool::ffn::ResultCode::ProtocolMismatch;
}

/// AtnAgent signal that the admitted Lane input is visible.
struct InputReady {
  /// Admitted invocation identity.
  InvocationKey key;
  /// Active Executor Lane lease sequence.
  std::uint64_t executor_lease_sequence;
  /// Return the Executor lease sequence used for publication.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return executor_lease_sequence; }
  /// Validate the shared Lane-scoped identity.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const { return validate_lane_success(*this); }
  constexpr bool operator==(const InputReady &) const = default;
};

/// FfnAgent signal that MoE routing metadata is visible to its TP peers.
struct RoutingMetadataReady {
  /// Admitted invocation identity.
  InvocationKey key;
  /// Active Executor Lane lease sequence.
  std::uint64_t executor_lease_sequence;
  /// Return the Executor lease sequence used for publication.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return executor_lease_sequence; }
  /// Validate the shared Lane-scoped identity.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const { return validate_lane_success(*this); }
  constexpr bool operator==(const RoutingMetadataReady &) const = default;
};

/// FfnAgent signal that its rank-local TP partial is visible.
struct PartialReady {
  /// Admitted invocation identity.
  InvocationKey key;
  /// Active Executor Lane lease sequence.
  std::uint64_t executor_lease_sequence;
  /// Return the Executor lease sequence used for publication.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return executor_lease_sequence; }
  /// Validate the shared Lane-scoped identity.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const { return validate_lane_success(*this); }
  constexpr bool operator==(const PartialReady &) const = default;
};

/// FfnAgent signal that compute and its selected delivery are complete.
struct FfnAgentCompletion {
  /// Admitted invocation identity.
  InvocationKey key;
  /// Active Executor Lane lease sequence.
  std::uint64_t executor_lease_sequence;
  /// Return the Executor lease sequence used for publication.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return executor_lease_sequence; }
  /// Validate the shared Lane-scoped identity.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const { return validate_lane_success(*this); }
  constexpr bool operator==(const FfnAgentCompletion &) const = default;
};

/// Validate the identity shared by successful Instance-scoped records.
/// \tparam Record Instance-scoped record containing `key`.
template <class Record>
XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate_instance_success(const Record &record) {
  return record.key.valid() ? xpool::ffn::ResultCode::Ok : xpool::ffn::ResultCode::ProtocolMismatch;
}

/// Coordinator signal committing a complete output for Instance consumption.
struct OutputCommit {
  /// Committed invocation identity.
  InvocationKey key;

  /// Return the invocation sequence used for publication.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return key.invocation_sequence; }
  /// Validate the shared Instance-scoped identity.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const { return validate_instance_success(*this); }
  constexpr bool operator==(const OutputCommit &) const = default;
};

/// AtnAgent signal acknowledging that a committed output was consumed.
struct OutputAcknowledgement {
  /// Acknowledged invocation identity.
  InvocationKey key;

  /// Return the invocation sequence used for publication.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t publication_sequence() const { return key.invocation_sequence; }
  /// Validate the shared Instance-scoped identity.
  XPOOL_HOST_DEVICE_FN constexpr xpool::ffn::ResultCode validate() const { return validate_instance_success(*this); }
  constexpr bool operator==(const OutputAcknowledgement &) const = default;
};

/// Canonical first-failure payload shared across all Fabric PEs.
struct FailurePayload {
  /// Non-Ok canonical result code.
  xpool::ffn::ResultCode result_code;
  /// PE that first claimed the failure slot.
  int origin_pe;
  /// Invocation active when the failure occurred.
  InvocationKey key;
  /// Config-order layer ordinal active at failure.
  std::size_t layer_ordinal;

  constexpr bool operator==(const FailurePayload &) const = default;
};

/// Generation-wide first-failure slot with claim and publication phases.
struct Failure {
  /// Atomic claim word; zero means no PE has claimed failure ownership.
  std::uint64_t claim;
  /// Publication word made visible after the winning payload is complete.
  std::uint64_t publication;
  /// Payload written only by the winning claimant.
  FailurePayload payload;

#if defined(__CUDACC__)
  /// Return whether a failure payload has been published locally.
  XPOOL_DEVICE_FN XPOOL_DEVICE_FORCEINLINE bool published() const;
  /// Claim and publish the first local failure when this PE wins.
  /// Only ProtocolMismatch and Timeout are admissible. The winning PE owns the
  /// payload; later contenders leave it unchanged.
  XPOOL_DEVICE_FN bool try_publish(int coordinator_pe, xpool::ffn::ResultCode result_code, const InvocationKey &key,
                                   std::size_t layer_ordinal);
  /// Replicate a published failure to every participating PE.
  XPOOL_DEVICE_FN void publish_to_all(int pe_count) const;
#endif
};

/// Trivially copyable record accepted by `Publication`.
/// \tparam Record Candidate record type with publication and validation operations.
template <class Record>
concept PublicationRecord = std::is_standard_layout_v<Record> && std::is_trivially_copyable_v<Record> &&
    requires(const Record &record) {
  { record.publication_sequence() } -> std::same_as<std::uint64_t>;
  { record.validate() } -> std::same_as<xpool::ffn::ResultCode>;
  { record.key } -> std::same_as<const InvocationKey &>;
};

/// One aligned record and its release-published monotonic sequence.
/// \tparam Record Trivially copyable Fabric protocol record.
template <PublicationRecord Record> struct alignas(kFabricPublicationAlignment) Publication {
  /// Record bytes published before `sequence` becomes visible.
  Record record;
  /// Monotonic publication sequence; zero denotes unpublished storage.
  std::uint64_t sequence;

#if defined(__CUDACC__)
  /// Test whether the publication has reached a minimum sequence.
  [[nodiscard]] XPOOL_DEVICE_FN bool test_at_least(std::uint64_t expected_sequence) const;
  /// Validate the currently published record against expected identity.
  [[nodiscard]] XPOOL_DEVICE_FN xpool::ffn::ResultCode validate_expected(std::uint64_t expected_sequence,
                                                                         const InvocationKey &expected_key) const;

  /// Publish record bytes and their release signal to one destination PE.
  XPOOL_DEVICE_FN void publish_record(const cooperative_groups::thread_block &group, int destination_pe);

  /// Copy one symmetric payload remotely, fence its visibility, then publish
  /// the matching record and release signal at the same PE.
  XPOOL_DEVICE_FN void publish_payload(const cooperative_groups::thread_block &group, int destination_pe,
                                       cuda::std::span<std::uint8_t> symmetric_payload);

  /// Release-publish after all block participants completed local payload writes.
  /// Any preceding NVSHMEM work completes before destination records signal
  /// that the locally addressed payload may be consumed.
  XPOOL_DEVICE_FN void publish_after_local_payload(const cooperative_groups::thread_block &group,
                                                   cuda::std::span<const int> destination_pes);

  /// Complete prior remote payload writes before release-publishing to one PE.
  XPOOL_DEVICE_FN void publish_after_remote_payload(const cooperative_groups::thread_block &group, int destination_pe);
#endif
};

static_assert(sizeof(Publication<Submission>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<Admission>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<LaneExecution>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<InputReady>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<RoutingMetadataReady>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<PartialReady>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<FfnAgentCompletion>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<OutputCommit>) == kFabricPublicationAlignment);
static_assert(sizeof(Publication<OutputAcknowledgement>) == kFabricPublicationAlignment);
static_assert(std::is_trivially_copyable_v<Failure>);

} // namespace xpool::fabric
