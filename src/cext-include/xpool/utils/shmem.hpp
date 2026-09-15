#pragma once

/// \file xpool/utils/shmem.hpp
/// \brief Move-only ownership for POSIX shared-memory mappings.

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace xpool::utils {

/// One process-local mapping of a POSIX shared-memory object.
///
/// A mapping created with create() also owns the shared-memory name and unlinks
/// it during cleanup. A mapping returned by attach() only releases its local
/// descriptor and mapping. Destruction is best-effort; call close() when POSIX
/// cleanup failures must be reported.
class PosixSharedMemoryMapping {
public:
  /// Construct an empty mapping owner.
  PosixSharedMemoryMapping() = default;

  /// Create, size, and map a new shared-memory object with an exact name.
  static PosixSharedMemoryMapping create(std::string name, std::size_t bytes);

  /// Open and map an existing shared-memory object.
  static PosixSharedMemoryMapping attach(std::string_view name);

  ~PosixSharedMemoryMapping() noexcept;

  PosixSharedMemoryMapping(const PosixSharedMemoryMapping &) = delete;
  PosixSharedMemoryMapping &operator=(const PosixSharedMemoryMapping &) = delete;

  /// Transfer descriptor, mapping, and optional unlink authority.
  PosixSharedMemoryMapping(PosixSharedMemoryMapping &&other) noexcept;

  /// Release this mapping before taking another mapping and its authority.
  PosixSharedMemoryMapping &operator=(PosixSharedMemoryMapping &&other) noexcept;

  /// Return the complete mapped byte range.
  std::span<std::byte> bytes() const noexcept { return mapping_; }

  /// Return the creator-owned name used for unlinking.
  /// \pre This mapping was returned by create() and has not been closed.
  std::string_view unlink_name() const noexcept { return *unlink_name_; }

  /// Unmap, close, and optionally unlink this object. Repeated calls are no-ops.
  void close();

private:
  void close(bool check_errors);

  int descriptor_ = -1;
  std::span<std::byte> mapping_;
  std::optional<std::string> unlink_name_;
};

} // namespace xpool::utils
