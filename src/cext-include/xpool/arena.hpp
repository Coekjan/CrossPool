#pragma once

/// \file xpool/arena.hpp
/// \brief Common allocation-root protocol for native CrossPool arenas.

#include <cstddef>
#include <cstdint>
#include <type_traits>

/// Shared native arena allocation protocol.
namespace xpool::arena {

/// Required alignment of a complete CUDA or NVSHMEM arena allocation.
inline constexpr std::size_t kAllocationAlignment = 256;

/// Alignment used by hidden-state payload regions and cooperative copies.
inline constexpr std::size_t kPayloadAlignment = 16;

/// Common prefix stored at offset zero of every concrete arena layout.
struct LayoutHeader {
  /// Domain-specific magic identifying the concrete arena kind.
  std::uint64_t magic;
  /// Native ABI version required to interpret the concrete layout.
  std::uint32_t abi_version;
  /// Size in bytes of the complete concrete arena layout.
  std::size_t layout_size;
  /// Total bytes occupied by the complete arena allocation.
  std::size_t total_bytes;
  /// Byte offset of the concrete arena's mutable state.
  std::size_t state_offset;

  constexpr bool operator==(const LayoutHeader &) const = default;
};

static_assert(std::is_trivially_copyable_v<LayoutHeader>);
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t), "xpool arena ABI requires 64-bit size_t");

} // namespace xpool::arena
