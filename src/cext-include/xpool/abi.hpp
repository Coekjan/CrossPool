#pragma once

/// \file xpool/abi.hpp
/// \brief Stable FFN values shared by Python and native xpool data planes.

#include <cstddef>
#include <cstdint>

#include <xpool/abort.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/enum.hpp>

/// Runtime-neutral values shared across xpool native protocol boundaries.
namespace xpool::abi {

/// Native ABI version checked when the extension is loaded and in wire data.
inline constexpr std::uint32_t kAbiVersion = 54;

/// Runtime forward mode values accepted by the xpool FFN shim ABI.
struct XPoolForwardMode {
  /// On-wire forward-mode integer values.
  enum Type : std::uint32_t {
    /// Extend/prefill-mode FFN request for a contiguous prompt-token batch.
    Extend = 1,
    /// Decode-mode FFN request for one scheduler decode step.
    Decode = 2,
    /// Data-parallel idle-rank request with no live FFN work.
    Idle = 4,
  };

  /// Construct a validated forward mode from its same-domain enum.
  /// \param value Stable forward-mode value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  XPOOL_HOST_DEVICE_FN
  constexpr XPoolForwardMode(Type value) : XPoolForwardMode(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated forward mode from an integer.
  /// \tparam T Non-boolean integral input type.
  /// \param value Stable forward-mode value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr XPoolForwardMode(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped forward-mode value.
  /// \return Stable value carried by this trusted wrapper.
  XPOOL_HOST_DEVICE_FN
  constexpr Type value() const { return value_; }

  /// Return whether an integer is a valid on-wire forward mode.
  /// \tparam T This type's Type or a non-boolean integral type.
  /// \param value Raw integer supplied by a producer.
  /// \return True for every declared forward mode.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, Extend) || xpool::utils::enum_value_equal(value, Decode) ||
           xpool::utils::enum_value_equal(value, Idle);
  }

  /// Compare two forward-mode wrappers.
  /// \return True when both wrappers carry the same value.
  constexpr bool operator==(const XPoolForwardMode &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Stable forward-mode value to compare.
  /// \return True when this wrapper carries value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// SGLang-facing handoff contract for one FFN result contribution.
struct FfnResultHandoff {
  /// On-wire result-handoff integer values.
  enum Type : std::uint32_t {
    /// Every result-group recipient receives the complete contribution.
    ReplicatedFull = 1,
    /// Rank zero receives the complete additive input for reduce-scatter.
    ReduceScatterInput = 2,
  };

  /// Construct a validated result handoff from its same-domain enum.
  /// \param value Stable result-handoff value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  XPOOL_HOST_DEVICE_FN
  constexpr FfnResultHandoff(Type value) : FfnResultHandoff(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated result handoff from an integer.
  /// \tparam T Non-boolean integral input type.
  /// \param value Stable result-handoff value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr FfnResultHandoff(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped result-handoff value.
  /// \return Stable value carried by this trusted wrapper.
  XPOOL_HOST_DEVICE_FN
  constexpr Type value() const { return value_; }

  /// Return whether an integer is a valid result handoff.
  /// \tparam T This type's Type or a non-boolean integral type.
  /// \param value Raw integer supplied by a producer.
  /// \return True for every declared result handoff.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, ReplicatedFull) ||
           xpool::utils::enum_value_equal(value, ReduceScatterInput);
  }

  /// Compare two result-handoff wrappers.
  /// \return True when both wrappers carry the same value.
  constexpr bool operator==(const FfnResultHandoff &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Stable result-handoff value to compare.
  /// \return True when this wrapper carries value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// Data-parallel padding mode represented in FFN request metadata.
struct DpPaddingMode {
  /// On-wire data-parallel padding-mode integer values.
  enum Type : std::uint32_t {
    /// Request is not running through data-parallel synchronization.
    None = 0,
    /// Producer padded each DP rank to the maximum rank-local token count.
    MaxLen = 1,
    /// Producer packed all DP rank token counts into one summed buffer.
    SumLen = 2,
  };

  /// Construct a validated padding mode from its same-domain enum.
  /// \param value Stable padding-mode value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  XPOOL_HOST_DEVICE_FN
  constexpr DpPaddingMode(Type value) : DpPaddingMode(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated padding mode from an integer.
  /// \tparam T Non-boolean integral input type.
  /// \param value Stable padding-mode value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr DpPaddingMode(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped padding-mode value.
  /// \return Stable value carried by this trusted wrapper.
  XPOOL_HOST_DEVICE_FN
  constexpr Type value() const { return value_; }

  /// Return whether an integer is a valid padding mode.
  /// \tparam T This type's Type or a non-boolean integral type.
  /// \param value Raw integer supplied by a producer.
  /// \return True for every declared padding mode.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, None) || xpool::utils::enum_value_equal(value, MaxLen) ||
           xpool::utils::enum_value_equal(value, SumLen);
  }

  /// Compare two padding-mode wrappers.
  /// \return True when both wrappers carry the same value.
  constexpr bool operator==(const DpPaddingMode &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Stable padding-mode value to compare.
  /// \return True when this wrapper carries value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// Element dtype of hidden-state tensors carried by FFN protocols.
struct TensorDType {
  /// On-wire hidden-state dtype integer values.
  enum Type : std::uint32_t {
    /// bfloat16 hidden states.
    Bf16 = 1,
    /// IEEE float16 hidden states.
    Fp16 = 2,
    /// IEEE float32 hidden states.
    Fp32 = 3,
  };

  /// Construct a validated dtype from its same-domain enum.
  /// \param value Stable dtype value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  XPOOL_HOST_DEVICE_FN
  constexpr TensorDType(Type value) : TensorDType(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated dtype from an integer.
  /// \tparam T Non-boolean integral input type.
  /// \param value Stable dtype value.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr TensorDType(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped dtype value.
  /// \return Stable value carried by this trusted wrapper.
  XPOOL_HOST_DEVICE_FN
  constexpr Type value() const { return value_; }

  /// Return whether an integer is a defined dtype value.
  /// \tparam T This type's Type or a non-boolean integral type.
  /// \param value Raw integer supplied by a producer.
  /// \return True for every declared dtype.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, Bf16) || xpool::utils::enum_value_equal(value, Fp16) ||
           xpool::utils::enum_value_equal(value, Fp32);
  }

  /// Return the hidden-state element width in bytes.
  /// \return Element width in bytes.
  /// \pre This wrapper contains a value accepted by is_valid().
  /// \post An invalid wrapped value aborts the host process or current kernel.
  XPOOL_HOST_DEVICE_FN
  constexpr std::size_t bytes() const {
    switch (value_) {
    case Bf16:
    case Fp16:
      return 2U;
    case Fp32:
      return 4U;
    }
    xpool::abort();
  }

  /// Compare two dtype wrappers.
  /// \return True when both wrappers carry the same value.
  constexpr bool operator==(const TensorDType &) const = default;

  /// Compare this wrapper with one on-wire enum value.
  /// \param value Stable dtype value to compare.
  /// \return True when this wrapper carries value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// Result codes shared by Transport and Fabric completion paths.
struct FfnResultCode {
  /// Device-visible FFN result code values.
  enum Type : std::uint32_t {
    /// Result completed successfully.
    Ok = 0,
    /// Shutdown failed an in-flight request.
    Shutdown = 1,
    /// A device-visible record violates the FFN protocol contract.
    ProtocolMismatch = 2,
    /// A bounded device protocol phase exceeded its liveness deadline.
    Timeout = 3,
    /// The requested valid execution path has not been implemented.
    NotImplemented = 4,
  };

  /// Construct a fail-closed protocol-mismatch result.
  XPOOL_HOST_DEVICE_FN
  constexpr FfnResultCode() = default;

  /// Construct a validated result code from its same-domain enum.
  /// \param value Stable result error code.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  XPOOL_HOST_DEVICE_FN
  constexpr FfnResultCode(Type value) : FfnResultCode(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated result code from an integer.
  /// \tparam T Non-boolean integral input type.
  /// \param value Stable result error code.
  /// \pre is_valid(value) is true; violation fail-stops the process or kernel.
  template <xpool::utils::EnumInput<Type> T>
  XPOOL_HOST_DEVICE_FN explicit constexpr FfnResultCode(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped result error code.
  /// \return Stable value carried by this trusted wrapper.
  XPOOL_HOST_DEVICE_FN
  constexpr Type value() const { return value_; }

  /// Return whether an integer is a defined FFN result error code.
  /// \tparam T This type's Type or a non-boolean integral type.
  /// \param value Raw integer supplied by a producer.
  /// \return True for every declared result error code.
  template <xpool::utils::EnumInput<Type> T> XPOOL_HOST_DEVICE_FN static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, Ok) || xpool::utils::enum_value_equal(value, Shutdown) ||
           xpool::utils::enum_value_equal(value, ProtocolMismatch) ||
           xpool::utils::enum_value_equal(value, Timeout) || xpool::utils::enum_value_equal(value, NotImplemented);
  }

  /// Compare two result-code wrappers.
  /// \return True when both wrappers carry the same value.
  constexpr bool operator==(const FfnResultCode &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Stable result error code to compare.
  /// \return True when this wrapper carries value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_ = ProtocolMismatch;
};

} // namespace xpool::abi
