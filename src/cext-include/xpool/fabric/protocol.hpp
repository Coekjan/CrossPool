#pragma once

/// \file xpool/fabric/protocol.hpp
/// \brief Typed NVSHMEM records for one distributed FFN invocation.

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <xpool/abi.hpp>
#include <xpool/abort.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/enum.hpp>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#endif

namespace xpool::fabric {

/// Data-plane identity of one model-level FFN invocation.
struct FfnInvocationKey {
  /// Config-order model identity.
  std::size_t model_index;
  /// Positive monotonic sequence jointly presented by all AtnAgents.
  std::uint64_t invocation_sequence;

  /// Return whether this key can identify a published invocation.
  /// \return True when the invocation sequence is positive.
  XPOOL_HOST_DEVICE_FN constexpr bool valid() const { return invocation_sequence != 0; }

  /// Compare two invocation identities.
  /// \return True when model index and sequence are equal.
  constexpr bool operator==(const FfnInvocationKey &) const = default;
};

/// Coordinator-derived execution mode for a complete model invocation.
struct FfnExecutionMode {
  /// Stable execution-mode values used by Fabric records.
  enum Type : std::uint32_t {
    /// Variable-size prompt-token execution through Executor-owned payloads.
    Prefill = 1,
    /// Fixed-capacity decode execution through model-owned payloads.
    Decode = 2,
  };

  /// Construct a validated execution mode from its same-domain enum.
  /// \param value Supported execution mode.
  XPOOL_HOST_DEVICE_FN constexpr FfnExecutionMode(Type value)
      : FfnExecutionMode(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated execution mode from an integer.
  /// \tparam T Same-domain enum or non-boolean integral input.
  /// \param value Candidate execution-mode representation.
  /// \pre is_valid(value) is true; violation fail-stops.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr FfnExecutionMode(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped execution mode.
  /// \return Stable execution-mode enum value.
  XPOOL_HOST_DEVICE_FN constexpr Type value() const { return value_; }

  /// Return whether a raw value names a supported execution mode.
  /// \tparam T Same-domain enum or non-boolean integral input.
  /// \param value Candidate execution-mode representation.
  /// \return True when value denotes Prefill or Decode.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, Prefill) || xpool::utils::enum_value_equal(value, Decode);
  }

  /// Compare two validated execution modes.
  /// \return True when both wrappers contain the same mode.
  constexpr bool operator==(const FfnExecutionMode &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Execution mode to compare.
  /// \return True when this wrapper contains value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// Complete model-level invocation derived by the Coordinator.
struct FfnInvocation {
  /// Model and sequence identity.
  FfnInvocationKey key;
  /// Config-order FFN layer ordinal.
  std::size_t layer_ordinal;
  /// Physical rows in the complete invocation payload.
  std::size_t payload_rows;
  /// AtnAgent PE selected as the deterministic Input Publisher.
  int input_pe;
  /// Raw FfnExecutionMode value.
  std::uint32_t execution_mode;
  /// Raw abi::FfnResultHandoff value.
  std::uint32_t result_handoff;
  /// Raw abi::DpPaddingMode value.
  std::uint32_t dp_padding_mode;

  /// Validate intrinsic invocation identity and closed-set values.
  /// \return Ok for a structurally valid invocation, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return key.valid() && payload_rows != 0 && input_pe >= 0 && FfnExecutionMode::is_valid(execution_mode) &&
                   xpool::abi::FfnResultHandoff::is_valid(result_handoff) &&
                   xpool::abi::DpPaddingMode::is_valid(dp_padding_mode)
               ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
               : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

  /// Validate this invocation against immutable arena and model geometry.
  /// \param layout Generation-wide participant and Executor geometry.
  /// \param model Model payload and layer geometry selected by key.model_index.
  /// \return Ok when intrinsic facts and contextual bounds are valid, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate(
      const FabricArenaLayout &layout, const FabricModelLayout &model) const {
    const auto invalid = xpool::abi::FfnResultCode{
        xpool::abi::FfnResultCode::ProtocolMismatch};
    if (validate() != xpool::abi::FfnResultCode::Ok ||
        key.model_index >= layout.model_count || layer_ordinal >= model.layer_count ||
        input_pe != 0 || model.hidden_size == 0 ||
        !xpool::abi::TensorDType::is_valid(model.dtype)) {
      return invalid;
    }
    const auto element_bytes = xpool::abi::TensorDType{model.dtype}.bytes();
    if (payload_rows > std::numeric_limits<std::size_t>::max() /
                           model.hidden_size / element_bytes) {
      return invalid;
    }
    const auto payload_bytes = payload_rows * model.hidden_size * element_bytes;
    const auto prefill = execution_mode == FfnExecutionMode::Prefill;
    const auto capacity = prefill ? model.prefill_payload_capacity_bytes
                                  : model.decode_payload_capacity_bytes;
    return payload_bytes <= capacity &&
                   (!prefill || payload_bytes <= layout.executor_payload_capacity_bytes)
               ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
               : invalid;
  }

  /// Compare two derived invocations.
  /// \return True when every invocation field is equal.
  constexpr bool operator==(const FfnInvocation &) const = default;
};

/// One AtnAgent's model-scoped invocation submission.
struct FfnSubmission {
  /// Model and sequence identity presented by this AtnAgent.
  FfnInvocationKey key;
  /// Config-order FFN layer ordinal.
  std::size_t layer_ordinal;
  /// Physical rows in this AtnAgent's replicated hidden-state buffer.
  std::size_t payload_rows;
  /// Rank-local live token count before DP padding; zero is valid for Idle.
  std::size_t local_token_count;
  /// Raw abi::XPoolForwardMode value.
  std::uint32_t forward_mode;
  /// Raw abi::FfnResultHandoff value.
  std::uint32_t result_handoff;
  /// Raw abi::DpPaddingMode value.
  std::uint32_t dp_padding_mode;

  /// Validate intrinsic submission identity and closed-set values.
  /// \return Ok for a structurally valid submission, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return key.valid() && payload_rows != 0 && local_token_count <= payload_rows &&
                   xpool::abi::XPoolForwardMode::is_valid(forward_mode) &&
                   xpool::abi::FfnResultHandoff::is_valid(result_handoff) &&
                   xpool::abi::DpPaddingMode::is_valid(dp_padding_mode)
               ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
               : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

  /// Compare two source submissions.
  /// \return True when every submission field is equal.
  constexpr bool operator==(const FfnSubmission &) const = default;
};

/// Coordinator-owned admission of one invocation to an Executor.
struct FfnExecutionAdmission {
  /// Admitted model and sequence identity.
  FfnInvocationKey key;
  /// Distributed Executor leased to the invocation.
  std::size_t executor_index;
  /// Raw FfnExecutionMode value derived by the Coordinator.
  std::uint32_t execution_mode;

  /// Validate intrinsic admission identity and execution mode.
  /// \return Ok for a structurally valid admission, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return key.valid() && FfnExecutionMode::is_valid(execution_mode)
               ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
               : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

  /// Compare two execution admissions.
  /// \return True when invocation, Executor, and mode are equal.
  constexpr bool operator==(const FfnExecutionAdmission &) const = default;
};

/// Input Publisher proof that one Prefill Executor payload is visible.
struct FfnInputReady {
  /// Invocation whose Executor-owned input is ready.
  FfnInvocationKey key;

  /// Validate this success-only record.
  /// \return Ok when the invocation key is valid, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return key.valid() ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
                       : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

  /// Compare two input-readiness records.
  /// \return True when both records identify the same invocation.
  constexpr bool operator==(const FfnInputReady &) const = default;
};

/// One FfnAgent's successful completion of an Executor invocation.
struct FfnAgentCompletion {
  /// Successfully completed invocation.
  FfnInvocationKey key;

  /// Validate this success-only record.
  /// \return Ok when the invocation key is valid, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return key.valid() ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
                       : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

  /// Compare two FfnAgent completion records.
  /// \return True when both records identify the same invocation.
  constexpr bool operator==(const FfnAgentCompletion &) const = default;
};

/// Mathematical contribution delivered to one AtnAgent recipient.
struct FfnResultContribution {
  /// Stable result-contribution values.
  enum Type : std::uint32_t {
    /// Complete mathematical FFN contribution.
    Full = 1,
    /// Additive zero, materialized locally without a payload transfer.
    Zero = 2,
  };

  /// Construct a validated contribution from its same-domain enum.
  /// \param value Supported mathematical contribution.
  XPOOL_HOST_DEVICE_FN constexpr FfnResultContribution(Type value)
      : FfnResultContribution(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated contribution from an integer.
  /// \tparam T Same-domain enum or non-boolean integral input.
  /// \param value Candidate contribution representation.
  /// \pre is_valid(value) is true; violation fail-stops.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr FfnResultContribution(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped contribution value.
  /// \return Stable contribution enum value.
  XPOOL_HOST_DEVICE_FN constexpr Type value() const { return value_; }

  /// Return whether a raw value names a supported contribution.
  /// \tparam T Same-domain enum or non-boolean integral input.
  /// \param value Candidate contribution representation.
  /// \return True when value denotes Full or Zero.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, Full) || xpool::utils::enum_value_equal(value, Zero);
  }

  /// Compare two validated result contributions.
  /// \return True when both wrappers contain the same contribution.
  constexpr bool operator==(const FfnResultContribution &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Contribution to compare.
  /// \return True when this wrapper contains value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// Coordinator-published successful contribution for one AtnAgent.
struct FfnResult {
  /// Completed model and sequence identity.
  FfnInvocationKey key;
  /// Raw FfnResultContribution value.
  std::uint32_t contribution;

  /// Validate this success-only result record.
  /// \return Ok for a valid key and contribution, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return key.valid() && FfnResultContribution::is_valid(contribution)
               ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
               : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

  /// Compare two successful result records.
  /// \return True when invocation and contribution are equal.
  constexpr bool operator==(const FfnResult &) const = default;
};

/// AtnAgent proof that one successful Fabric result is no longer read.
struct FfnResultAcknowledgement {
  /// Acknowledged model and sequence identity.
  FfnInvocationKey key;

  /// Validate this success-only acknowledgement.
  /// \return Ok when the invocation key is valid, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return key.valid() ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
                       : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

  /// Compare two result acknowledgements.
  /// \return True when both records identify the same invocation.
  constexpr bool operator==(const FfnResultAcknowledgement &) const = default;
};

/// Immutable payload of the first canonical invocation failure.
struct FabricFailurePayload {
  /// Raw abi::FfnResultCode value eligible for canonical publication.
  std::uint32_t result_code;
  /// NVSHMEM PE that first claimed the failure.
  int origin_pe;
  /// Failed model and invocation sequence.
  FfnInvocationKey key;
  /// Config-order FFN layer ordinal associated with the failure.
  std::size_t layer_ordinal;

  /// Compare two immutable failure payloads.
  /// \return True when every canonical failure field is equal.
  constexpr bool operator==(const FabricFailurePayload &) const = default;
};

/// First-writer-wins canonical Fabric invocation failure.
struct FabricFailure {
  /// Coordinator-local remote-CAS arbitration word.
  std::uint64_t claim;
  /// PE-local visibility signal; one means payload is published.
  std::uint64_t publication;
  /// Immutable payload written by the unique successful claimant.
  FabricFailurePayload payload;

#if defined(__CUDACC__)
  /// Return whether this PE has acquired the canonical failure payload.
  /// \return True after local release publication of the payload.
  XPOOL_DEVICE_FN bool published() const;

  /// Try to claim and publish one eligible canonical failure.
  /// \param coordinator_pe NVSHMEM PE that owns the arbitration word.
  /// \param result_code Eligible canonical failure code.
  /// \param key Failed invocation identity.
  /// \param layer_ordinal Model-local layer ordinal associated with the failure.
  /// \return True only for the unique successful claimant.
  XPOOL_DEVICE_FN bool try_publish(int coordinator_pe, xpool::abi::FfnResultCode result_code,
                                   const FfnInvocationKey &key, std::size_t layer_ordinal);

  /// Fan the Coordinator's canonical failure payload and signal to all PEs.
  /// \param pe_count Total number of Fabric PEs.
  XPOOL_DEVICE_FN void publish_to_all(int pe_count) const;
#endif
};

/// Fixed-size record suitable for one typed Fabric publication.
template <class Record>
concept FabricRecord = std::is_standard_layout_v<Record> && std::is_trivially_copyable_v<Record> &&
                       requires(const Record &record) {
                         { record.validate() } -> std::same_as<xpool::abi::FfnResultCode>;
                         { record.key.invocation_sequence } -> std::convertible_to<std::uint64_t>;
                       };

/// One symmetric typed record followed by its publication sequence.
template <FabricRecord Record>
struct alignas(kFabricPublicationAlignment) FabricPublication {
  /// Record bytes written before sequence publication.
  Record record;
  /// Zero for an empty Executor publication or a positive model sequence.
  std::uint64_t sequence;

  /// Validate record identity against the published sequence.
  /// \return Ok for a complete matching publication, otherwise ProtocolMismatch.
  XPOOL_HOST_DEVICE_FN constexpr xpool::abi::FfnResultCode validate() const {
    return sequence != 0 && sequence == record.key.invocation_sequence &&
                   record.validate() == xpool::abi::FfnResultCode::Ok
               ? xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok}
               : xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  }

#if defined(__CUDACC__)
  /// Acquire-observe this PE's publication sequence.
  /// \return Zero for empty, or the positive published invocation sequence.
  XPOOL_DEVICE_FN std::uint64_t observe() const;

  /// Publish record and sequence to one PE with warp participation.
  /// \param group Warp tile whose threads participate in publication.
  /// \param sequence Positive invocation sequence copied into the signal.
  /// \param destination_pe Destination NVSHMEM PE.
  XPOOL_DEVICE_FN void publish(const cooperative_groups::thread_block_tile<32> &group, std::uint64_t sequence,
                               int destination_pe);

  /// Publish payload, record, and sequence to one PE with warp participation.
  /// \param group Warp tile whose threads participate in publication.
  /// \param sequence Positive invocation sequence copied into the signal.
  /// \param destination_pe Destination NVSHMEM PE.
  /// \param payload Symmetric local source whose peer address receives the payload.
  /// \param payload_bytes Number of payload bytes copied before publication.
  XPOOL_DEVICE_FN void publish(const cooperative_groups::thread_block_tile<32> &group, std::uint64_t sequence,
                               int destination_pe, void *payload, std::size_t payload_bytes);

  /// Publish record and sequence to one PE with block participation.
  /// \param group Thread block whose threads participate in publication.
  /// \param sequence Positive invocation sequence copied into the signal.
  /// \param destination_pe Destination NVSHMEM PE.
  XPOOL_DEVICE_FN void publish(const cooperative_groups::thread_block &group, std::uint64_t sequence,
                               int destination_pe);

  /// Publish payload, record, and sequence to one PE with block participation.
  /// \param group Thread block whose threads participate in publication.
  /// \param sequence Positive invocation sequence copied into the signal.
  /// \param destination_pe Destination NVSHMEM PE.
  /// \param payload Symmetric local source whose peer address receives the payload.
  /// \param payload_bytes Number of payload bytes copied before publication.
  XPOOL_DEVICE_FN void publish(const cooperative_groups::thread_block &group, std::uint64_t sequence,
                               int destination_pe, void *payload, std::size_t payload_bytes);

  /// Clear this PE's Executor-scoped publication before reuse.
  XPOOL_DEVICE_FN void clear();
#endif
};

static_assert(std::is_trivially_copyable_v<FabricFailure>);

} // namespace xpool::fabric
