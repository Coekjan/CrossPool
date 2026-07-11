#pragma once

/// \file xpool/utils/hex.hpp
/// \brief Host-side lowercase hexadecimal encoding utilities.

#include <c10/util/Exception.h>

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

/// Lowercase hexadecimal codecs used by xpool native host code.
namespace xpool::utils::hex {

/// Encode bytes as lowercase hexadecimal text.
/// \param payload Pointer to byte-addressable input data.
/// \param byte_count Number of bytes to encode.
/// \return Lowercase hexadecimal text with two characters per byte.
/// \pre payload points to at least byte_count bytes when byte_count is nonzero.
inline std::string encode_lowercase(const void *payload,
                                    std::size_t byte_count) {
  const auto *bytes = static_cast<const unsigned char *>(payload);
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < byte_count; ++index) {
    stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return stream.str();
}

/// Decode lowercase hexadecimal text into bytes.
/// \param text Lowercase hexadecimal text with two characters per byte.
/// \return Decoded bytes.
/// \throws c10::Error if the text has odd length or contains non-lowercase
/// hexadecimal characters.
inline std::vector<unsigned char> decode_lowercase(std::string_view text) {
  TORCH_CHECK(text.size() % 2 == 0,
              "xpool lowercase hex text must have an even length");
  std::vector<unsigned char> bytes(text.size() / 2);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    auto decode = [](char value) -> unsigned int {
      if (value >= '0' && value <= '9') {
        return static_cast<unsigned int>(value - '0');
      }
      if (value >= 'a' && value <= 'f') {
        return static_cast<unsigned int>(value - 'a' + 10);
      }
      TORCH_CHECK(false, "xpool lowercase hex text contains an invalid byte");
    };
    char high = text[index * 2];
    char low = text[index * 2 + 1];
    bytes[index] =
        static_cast<unsigned char>((decode(high) << 4U) | decode(low));
  }
  return bytes;
}

} // namespace xpool::utils::hex
