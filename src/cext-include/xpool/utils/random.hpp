#pragma once

/// \file xpool/utils/random.hpp
/// \brief Dependency-light deterministic random utilities.

#include <cstdint>

#include <xpool/macros.hpp>

namespace xpool::utils::random {

/// Small deterministic SplitMix64 generator usable by Host and Device code.
class SplitMix64 {
public:
  /// Initialize the generator state.
  /// \param seed Initial 64-bit state.
  XPOOL_HOST_DEVICE_FN explicit constexpr SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

  /// Advance the generator and return the next value.
  /// \return Next deterministic 64-bit value.
  XPOOL_HOST_DEVICE_FN constexpr std::uint64_t next() noexcept {
    state_ += 0x9e3779b97f4a7c15ULL;
    auto value = state_;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

private:
  std::uint64_t state_;
};

} // namespace xpool::utils::random
