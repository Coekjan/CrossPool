#pragma once

/// \file xpool/utils/hex.hpp
/// \brief Host-side lowercase hexadecimal encoding utilities.

#include <c10/util/Exception.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

/// Lowercase hexadecimal codecs used by xpool native host code.
namespace xpool::utils::hex {

/// Encode bytes as lowercase hexadecimal text.
/// \param payload Pointer to byte-addressable input data.
/// \param byte_count Number of bytes to encode.
/// \return Lowercase hexadecimal text with two characters per byte.
/// \pre payload points to at least byte_count bytes when byte_count is nonzero.
std::string encode_lowercase(const void *payload, std::size_t byte_count);

/// Decode lowercase hexadecimal text into bytes.
/// \param text Lowercase hexadecimal text with two characters per byte.
/// \return Decoded bytes.
/// \throws c10::Error if the text has odd length or contains non-lowercase
/// hexadecimal characters.
std::vector<unsigned char> decode_lowercase(std::string_view text);

/// Strong opaque byte value with canonical lowercase hexadecimal projection.
/// \tparam T Trivially copyable native value whose complete object bytes form
/// its identity.
template <typename T>
  requires std::is_trivially_copyable_v<T>
class HexValue {
public:
  /// Construct a zero-initialized value.
  HexValue() = default;
  /// Construct from one complete native value.
  /// \param value Native bytes to own.
  explicit HexValue(T value) : value_(value) {}

  /// Decode one canonical lowercase hexadecimal value.
  /// \param text Lowercase hexadecimal text encoding exactly sizeof(T) bytes.
  /// \return Decoded strong value.
  static HexValue decode(std::string_view text) {
    const auto bytes = decode_lowercase(text);
    TORCH_CHECK(bytes.size() == sizeof(T), "xpool hexadecimal value has unexpected byte length");
    auto result = HexValue{};
    std::memcpy(&result.value_, bytes.data(), sizeof(T));
    return result;
  }

  /// Encode the complete native value as canonical lowercase hexadecimal.
  /// \return Two lowercase hexadecimal characters per native byte.
  std::string encode() const { return encode_lowercase(&value_, sizeof(T)); }

  /// Return mutable access for native APIs that initialize opaque storage.
  /// \return Owned native value.
  T &value() noexcept { return value_; }
  /// Return immutable access to the owned native value.
  /// \return Owned native value.
  const T &value() const noexcept { return value_; }

  /// Compare complete opaque object bytes for identity.
  /// \param other Strong value to compare.
  /// \return True when every object byte is equal.
  bool operator==(const HexValue &other) const noexcept {
    return std::memcmp(&value_, &other.value_, sizeof(T)) == 0;
  }
  /// Compare complete opaque object bytes for ordered-container placement.
  /// \param other Strong value to compare.
  /// \return True when this value precedes other lexicographically by byte.
  bool operator<(const HexValue &other) const noexcept {
    return std::memcmp(&value_, &other.value_, sizeof(T)) < 0;
  }

private:
  T value_{};
};

} // namespace xpool::utils::hex
