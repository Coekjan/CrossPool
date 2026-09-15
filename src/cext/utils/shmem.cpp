#include <xpool/utils/shmem.hpp>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <c10/util/Exception.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace xpool::utils {

PosixSharedMemoryMapping PosixSharedMemoryMapping::create(std::string name, std::size_t bytes) {
  TORCH_CHECK(!name.empty() && name.front() == '/', "xpool shared-memory name must start with '/'");
  TORCH_CHECK(bytes != 0 && bytes <= static_cast<std::size_t>(std::numeric_limits<off_t>::max()),
              "xpool shared-memory byte extent is invalid");

  auto result = PosixSharedMemoryMapping{};
  result.unlink_name_ = std::move(name);
  result.descriptor_ = shm_open(result.unlink_name_->c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (result.descriptor_ < 0) {
    const auto error = errno;
    result.unlink_name_.reset();
    TORCH_CHECK(false, "xpool shm_open failed: ", std::strerror(error));
  }
  TORCH_CHECK(ftruncate(result.descriptor_, static_cast<off_t>(bytes)) == 0,
              "xpool ftruncate failed: ", std::strerror(errno));
  auto *mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, result.descriptor_, 0);
  TORCH_CHECK(mapping != MAP_FAILED, "xpool mmap failed: ", std::strerror(errno));
  result.mapping_ = {static_cast<std::byte *>(mapping), bytes};
  std::memset(result.mapping_.data(), 0, result.mapping_.size());
  return result;
}

PosixSharedMemoryMapping PosixSharedMemoryMapping::attach(std::string_view name) {
  TORCH_CHECK(!name.empty() && name.front() == '/', "xpool shared-memory name must start with '/'");

  auto result = PosixSharedMemoryMapping{};
  const auto owned_name = std::string{name};
  result.descriptor_ = shm_open(owned_name.c_str(), O_RDWR, 0);
  TORCH_CHECK(result.descriptor_ >= 0, "xpool shm_open attach failed: ", std::strerror(errno));

  struct stat status{};
  TORCH_CHECK(fstat(result.descriptor_, &status) == 0, "xpool fstat failed: ", std::strerror(errno));
  TORCH_CHECK(status.st_size > 0, "xpool shared-memory object is empty");
  const auto bytes = static_cast<std::size_t>(status.st_size);
  auto *mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, result.descriptor_, 0);
  TORCH_CHECK(mapping != MAP_FAILED, "xpool mmap attach failed: ", std::strerror(errno));
  result.mapping_ = {static_cast<std::byte *>(mapping), bytes};
  return result;
}

PosixSharedMemoryMapping::~PosixSharedMemoryMapping() noexcept { close(false); }

PosixSharedMemoryMapping::PosixSharedMemoryMapping(PosixSharedMemoryMapping &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, -1)),
      mapping_(std::exchange(other.mapping_, std::span<std::byte>{})),
      unlink_name_(std::exchange(other.unlink_name_, std::nullopt)) {}

PosixSharedMemoryMapping &PosixSharedMemoryMapping::operator=(PosixSharedMemoryMapping &&other) noexcept {
  if (this != &other) {
    close(false);
    descriptor_ = std::exchange(other.descriptor_, -1);
    mapping_ = std::exchange(other.mapping_, std::span<std::byte>{});
    unlink_name_ = std::exchange(other.unlink_name_, std::nullopt);
  }
  return *this;
}

void PosixSharedMemoryMapping::close() { close(true); }

void PosixSharedMemoryMapping::close(bool check_errors) {
  auto error = 0;
  auto operation = static_cast<const char *>(nullptr);

  if (!mapping_.empty()) {
    if (munmap(mapping_.data(), mapping_.size()) != 0) {
      error = errno;
      operation = "munmap";
    }
    mapping_ = {};
  }
  if (descriptor_ >= 0) {
    if (::close(descriptor_) != 0) {
      if (error == 0) {
        error = errno;
        operation = "close";
      }
    }
    descriptor_ = -1;
  }
  if (unlink_name_) {
    if (shm_unlink(unlink_name_->c_str()) != 0) {
      if (error == 0) {
        error = errno;
        operation = "shm_unlink";
      }
    }
    unlink_name_.reset();
  }

  TORCH_CHECK(!check_errors || error == 0, "xpool ", operation, " failed: ", std::strerror(error));
}

} // namespace xpool::utils
