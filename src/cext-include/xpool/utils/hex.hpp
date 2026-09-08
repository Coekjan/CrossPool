#pragma once

/// \file xpool/utils/hex.hpp
/// \brief Host-side lowercase hexadecimal encoding utilities.

#include <c10/util/Exception.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>

/// Lowercase hexadecimal codecs used by CrossPool native host code.
namespace xpool::utils::hex {

/// Strong opaque byte value with canonical lowercase hexadecimal projection.
/// The complete object representation of T forms its identity.
template <typename T>
  requires std::is_trivially_copyable_v<T>
class HexValue {
public:
  /// Character count of the canonical lowercase hexadecimal projection.
  static constexpr std::size_t encoded_size = sizeof(T) * 2;

  /// Construct a zero-initialized value.
  HexValue() = default;
  /// Construct from one complete native value.
  explicit HexValue(T value) : value_(value) {}

  /// Decode one canonical lowercase hexadecimal value.
  /// \throws c10::Error for incorrect length or a non-lowercase hexadecimal digit.
  static HexValue decode(std::string_view text) {
    TORCH_CHECK(text.size() == encoded_size, "xpool hexadecimal value has unexpected byte length");
    auto result = HexValue{};
    auto *bytes = reinterpret_cast<unsigned char *>(&result.value_);
    const auto decode_nibble = [](char value) -> unsigned int {
      if (value >= '0' && value <= '9') {
        return static_cast<unsigned int>(value - '0');
      }
      if (value >= 'a' && value <= 'f') {
        return static_cast<unsigned int>(value - 'a' + 10);
      }
      TORCH_CHECK(false, "xpool lowercase hex text contains an invalid byte");
    };
    for (auto index = std::size_t{0}; index < sizeof(T); ++index) {
      bytes[index] = static_cast<unsigned char>((decode_nibble(text[index * 2]) << 4U) |
                                                decode_nibble(text[index * 2 + 1]));
    }
    return result;
  }

  /// Encode the complete native value as canonical lowercase hexadecimal.
  std::string encode() const {
    static constexpr auto digits = std::string_view{"0123456789abcdef"};
    const auto *bytes = reinterpret_cast<const unsigned char *>(&value_);
    auto result = std::string(encoded_size, '0');
    for (auto index = std::size_t{0}; index < sizeof(T); ++index) {
      result[index * 2] = digits[bytes[index] >> 4U];
      result[index * 2 + 1] = digits[bytes[index] & 0x0fU];
    }
    return result;
  }

  /// Return mutable access for native APIs that initialize opaque storage.
  T &value() noexcept { return value_; }
  /// Return immutable access to the owned native value.
  const T &value() const noexcept { return value_; }

  /// Compare complete opaque object bytes for identity.
  bool operator==(const HexValue &other) const noexcept {
    return std::memcmp(&value_, &other.value_, sizeof(T)) == 0;
  }
  /// Compare complete opaque object bytes for ordered-container placement.
  bool operator<(const HexValue &other) const noexcept {
    return std::memcmp(&value_, &other.value_, sizeof(T)) < 0;
  }

private:
  T value_{};
};

} // namespace xpool::utils::hex
