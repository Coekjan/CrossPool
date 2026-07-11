#pragma once

/// \file xpool/utils/layout.hpp
/// \brief Host-side helpers for offset-based memory layout construction.

#include <c10/util/Exception.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <xpool/utils/arith.hpp>

/// Generic helpers for constructing and validating in-memory byte layouts.
namespace xpool::utils::layout {

/// Named byte region inside an offset-based layout.
struct LayoutRegion {
  /// Human-readable region name used in validation diagnostics.
  const char *name;
  /// Byte offset of the region from the containing allocation base.
  std::int64_t offset;
  /// Byte count occupied by the region.
  std::int64_t bytes;
  /// Return whether two regions have identical names, offsets, and sizes.
  /// \return True when every region field matches exactly.
  bool operator==(const LayoutRegion &) const = default;
};

/// Declarative byte region specification used to build an aligned layout plan.
struct LayoutRegionSpec {
  /// Human-readable region name used in validation diagnostics.
  const char *name;
  /// Byte count requested before layout-level alignment is applied.
  std::int64_t byte_count;

  /// Describe storage for a raw byte region.
  /// \param name Human-readable region name.
  /// \param byte_count Requested byte count.
  /// \return Region specification for byte_count raw bytes.
  static LayoutRegionSpec bytes(const char *name, std::int64_t byte_count) {
    return LayoutRegionSpec{name, byte_count};
  }

  /// Describe storage for one object.
  /// \param name Human-readable region name.
  /// \return Region specification for one object of type T.
  template <typename T> static LayoutRegionSpec object(const char *name) {
    return LayoutRegionSpec{name, static_cast<std::int64_t>(sizeof(T))};
  }

  /// Describe storage for a contiguous object array.
  /// \param name Human-readable region name.
  /// \param count Number of array elements.
  /// \return Region specification for count objects of type T.
  /// \throws c10::Error if count is negative or size arithmetic overflows.
  template <typename T>
  static LayoutRegionSpec array(const char *name, std::int64_t count) {
    return bytes(
        name, XPOOL_CHECKED_MUL(count, static_cast<std::int64_t>(sizeof(T))));
  }
};

/// Concrete layout plan derived from an ordered region specification.
/// \tparam N Number of regions in the plan.
template <std::size_t N> struct LayoutPlan {
  /// Concrete regions with aligned offsets and unaligned byte sizes.
  std::array<LayoutRegion, N> regions;
  /// Total byte size of the containing allocation after final alignment.
  std::int64_t total_bytes;

  /// Build a concrete aligned layout plan from ordered region specifications.
  /// \param specs Ordered region specifications.
  /// \param alignment Positive byte alignment applied to every region
  /// boundary.
  /// \throws c10::Error if names are null, sizes are negative, alignment is
  /// invalid, or offset arithmetic overflows.
  explicit LayoutPlan(const std::array<LayoutRegionSpec, N> &specs,
                      std::int64_t alignment);

  /// Return a concrete region by index.
  /// \param index Region index in the original specification order.
  /// \return Region at index.
  const LayoutRegion &operator[](std::size_t index) const {
    return regions[index];
  }
};

template <std::size_t N>
LayoutPlan<N>::LayoutPlan(const std::array<LayoutRegionSpec, N> &specs,
                          std::int64_t alignment)
    : regions{}, total_bytes(0) {
  TORCH_CHECK(alignment > 0, "xpool layout plan alignment must be positive");
  std::int64_t offset = 0;
  for (std::size_t index = 0; index < N; ++index) {
    const LayoutRegionSpec &spec = specs[index];
    TORCH_CHECK(spec.name != nullptr,
                "xpool layout region name must not be null");
    TORCH_CHECK(spec.byte_count >= 0, "xpool layout region ", spec.name,
                " byte size must be non-negative");
    regions[index] = LayoutRegion{spec.name, offset, spec.byte_count};
    offset =
        XPOOL_CHECKED_ADD(offset, XPOOL_ALIGN_UP(spec.byte_count, alignment));
  }
  total_bytes = XPOOL_ALIGN_UP(offset, alignment);
}

} // namespace xpool::utils::layout
