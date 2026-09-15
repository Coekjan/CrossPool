#pragma once

/// \file xpool/utils/layout.hpp
/// \brief Host-side helpers for offset-based memory layout construction.

#include <array>
#include <cstddef>
#include <string_view>

#include <xpool/utils/checked.hpp>

/// Generic helpers for constructing and validating in-memory byte layouts.
namespace xpool::utils::layout {

/// Named byte region inside an offset-based layout.
struct LayoutRegion {
  /// Human-readable region name used in validation diagnostics.
  std::string_view name;
  /// Byte offset of the region from the containing allocation base.
  std::size_t offset;
  /// Byte count occupied by the region.
  std::size_t bytes;
  /// Required byte alignment of the region start.
  std::size_t alignment;
  bool operator==(const LayoutRegion &) const = default;
};

/// Declarative byte region specification used to build an aligned layout plan.
struct LayoutRegionSpec {
  /// Human-readable region name used in validation diagnostics.
  std::string_view name;
  /// Byte count requested before layout-level alignment is applied.
  std::size_t byte_count;
  /// Required byte alignment of the region start.
  std::size_t alignment;

  /// Describe storage for a raw byte region.
  static LayoutRegionSpec bytes(std::string_view name, std::size_t byte_count, std::size_t alignment = 1) {
    return LayoutRegionSpec{name, byte_count, alignment};
  }

  /// Describe storage for one object.
  template <typename T> static LayoutRegionSpec object(std::string_view name) {
    return LayoutRegionSpec{name, sizeof(T), alignof(T)};
  }

  /// Describe storage for a contiguous object array.
  /// \throws c10::Error if size arithmetic overflows.
  template <typename T> static LayoutRegionSpec array(std::string_view name, std::size_t count) {
    const auto byte_count = xpool::utils::checked::prod(count, std::size_t{sizeof(T)});
    return bytes(name, byte_count, alignof(T));
  }
};

/// Concrete layout plan derived from an ordered region specification.
template <std::size_t N> struct LayoutPlan {
  /// Concrete regions with aligned offsets and unaligned byte sizes.
  std::array<LayoutRegion, N> regions;
  /// Total byte size of the containing allocation after final alignment.
  std::size_t total_bytes;

  /// Build a concrete aligned layout plan from ordered region specifications.
  /// \throws c10::Error if a name is empty, alignment is zero, or offset
  /// arithmetic overflows.
  explicit LayoutPlan(const std::array<LayoutRegionSpec, N> &specs, std::size_t allocation_alignment);

  /// Return a concrete region by index.
  const LayoutRegion &operator[](std::size_t index) const { return regions[index]; }
};

template <std::size_t N>
LayoutPlan<N>::LayoutPlan(const std::array<LayoutRegionSpec, N> &specs, std::size_t allocation_alignment)
    : regions{}, total_bytes(0) {
  auto offset = std::size_t{0};
  for (std::size_t index = 0; index < N; ++index) {
    const auto &spec = specs[index];
    TORCH_CHECK(!spec.name.empty(), "xpool layout region name must not be empty");
    offset = xpool::utils::checked::align_up(offset, spec.alignment);
    regions[index] = LayoutRegion{spec.name, offset, spec.byte_count, spec.alignment};
    offset = xpool::utils::checked::sum(offset, spec.byte_count);
  }
  total_bytes = xpool::utils::checked::align_up(offset, allocation_alignment);
}

} // namespace xpool::utils::layout
