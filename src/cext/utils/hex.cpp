#include <xpool/utils/hex.hpp>

#include <c10/util/Exception.h>

#include <iomanip>
#include <sstream>

namespace xpool::utils::hex {

std::string encode_lowercase(const void *payload, std::size_t byte_count) {
  const auto *bytes = static_cast<const unsigned char *>(payload);
  auto stream = std::ostringstream{};
  stream << std::hex << std::setfill('0');
  for (auto index = std::size_t{0}; index < byte_count; ++index) {
    stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return stream.str();
}

std::vector<unsigned char> decode_lowercase(std::string_view text) {
  TORCH_CHECK(text.size() % 2 == 0, "xpool lowercase hex text must have an even length");
  auto bytes = std::vector<unsigned char>(text.size() / 2);
  const auto decode = [](char value) -> unsigned int {
    if (value >= '0' && value <= '9') {
      return static_cast<unsigned int>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<unsigned int>(value - 'a' + 10);
    }
    TORCH_CHECK(false, "xpool lowercase hex text contains an invalid byte");
  };
  for (auto index = std::size_t{0}; index < bytes.size(); ++index) {
    const auto high = text[index * 2];
    const auto low = text[index * 2 + 1];
    bytes[index] = static_cast<unsigned char>((decode(high) << 4U) | decode(low));
  }
  return bytes;
}

} // namespace xpool::utils::hex
