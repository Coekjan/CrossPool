/// \file tests/suites/cext/transport/shutdown_test.cu
/// \brief Process-wide cooperative Transport Resident shutdown tests.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>

#include <xpool/abi.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/transport/arena.hpp>
#include <xpool/transport/atnagent.hpp>
#include <xpool/utils/device.hpp>

namespace {

class TransportResident {
public:
  explicit TransportResident(
      std::span<const xpool::transport::TransportArenaView> arenas,
      xpool::fabric::FabricArenaView fabric_arena = {}) {
    TORCH_CHECK(!arenas.empty(),
                "Transport Resident test requires at least one arena");
    auto cuda_device = int{0};
    C10_CUDA_CHECK(cudaGetDevice(&cuda_device));
    cuda_device_ = static_cast<c10::DeviceIndex>(cuda_device);

    try {
      auto *arena_allocation = static_cast<void *>(nullptr);
      C10_CUDA_CHECK(cudaMalloc(&arena_allocation, arenas.size_bytes()));
      arenas_ = static_cast<xpool::transport::TransportArenaView *>(
          arena_allocation);
      C10_CUDA_CHECK(cudaMemcpy(arenas_, arenas.data(), arenas.size_bytes(),
                                cudaMemcpyHostToDevice));

      auto *state_allocation = static_cast<void *>(nullptr);
      C10_CUDA_CHECK(cudaMalloc(
          &state_allocation,
          sizeof(xpool::transport::TransportResidentState)));
      state_ = static_cast<xpool::transport::TransportResidentState *>(
          state_allocation);
      C10_CUDA_CHECK(cudaMemset(
          state_, 0, sizeof(xpool::transport::TransportResidentState)));

      control_stream_ = xpool::utils::device::OwnedCudaStream::create();
      resident_stream_ = xpool::utils::device::OwnedCudaStream::create();
      xpool::transport::launch_transport_resident_kernel(
          arenas_, arenas.size(), fabric_arena, state_, resident_stream_.get());
      launched_ = true;
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~TransportResident() { cleanup(); }

  TransportResident(const TransportResident &) = delete;
  TransportResident &operator=(const TransportResident &) = delete;
  TransportResident(TransportResident &&) = delete;
  TransportResident &operator=(TransportResident &&) = delete;

  void drain_async() {
    TORCH_CHECK(launched_ && state_ != nullptr,
                "Transport Resident test is not active");
    if (drain_requested_) {
      return;
    }
    const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
    auto *address = reinterpret_cast<std::uint32_t *>(
        reinterpret_cast<std::uint8_t *>(state_) +
        offsetof(xpool::transport::TransportResidentState, drain_requested));
    control_stream_.write_value(address, 1);
    drain_requested_ = true;
  }

  bool wait(std::chrono::milliseconds timeout = std::chrono::seconds{5}) const {
    const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (control_stream_.query() && resident_stream_.query()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return false;
  }

  void release() {
    TORCH_CHECK(state_ != nullptr,
                "Transport Resident test is already released");
    TORCH_CHECK(drain_requested_,
                "Transport Resident test must be drained before release");
    const auto device_guard = c10::cuda::CUDAGuard{cuda_device_};
    TORCH_CHECK(control_stream_.query() && resident_stream_.query(),
                "Transport Resident test cannot release pending resources");
    control_stream_.destroy();
    resident_stream_.destroy();
    C10_CUDA_CHECK(cudaFree(arenas_));
    arenas_ = nullptr;
    C10_CUDA_CHECK(cudaFree(state_));
    state_ = nullptr;
    launched_ = false;
  }

private:
  void cleanup() noexcept {
    if (state_ == nullptr && arenas_ == nullptr) {
      return;
    }
    C10_CUDA_IGNORE_ERROR(cudaSetDevice(cuda_device_));
    if (launched_ && !drain_requested_ && state_ != nullptr &&
        control_stream_) {
      auto *address = reinterpret_cast<std::uint32_t *>(
          reinterpret_cast<std::uint8_t *>(state_) +
          offsetof(xpool::transport::TransportResidentState,
                   drain_requested));
      static_cast<void>(control_stream_.try_write_value(address, 1));
    }
    if (resident_stream_) {
      C10_CUDA_IGNORE_ERROR(cudaStreamSynchronize(resident_stream_.get()));
    }
    if (control_stream_) {
      C10_CUDA_IGNORE_ERROR(cudaStreamSynchronize(control_stream_.get()));
    }
    if (arenas_ != nullptr) {
      C10_CUDA_IGNORE_ERROR(cudaFree(arenas_));
      arenas_ = nullptr;
    }
    if (state_ != nullptr) {
      C10_CUDA_IGNORE_ERROR(cudaFree(state_));
      state_ = nullptr;
    }
  }

  c10::DeviceIndex cuda_device_ = 0;
  xpool::transport::TransportArenaView *arenas_ = nullptr;
  xpool::transport::TransportResidentState *state_ = nullptr;
  xpool::utils::device::OwnedCudaStream control_stream_;
  xpool::utils::device::OwnedCudaStream resident_stream_;
  bool launched_ = false;
  bool drain_requested_ = false;
};

bool wait_for_mailbox_status(
    const xpool::transport::TransportArena &arena,
    xpool::transport::MailboxStatus expected,
    std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (arena.mailbox_status() == expected) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return false;
}

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class TransportShutdownCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto device_count = int{0};
    const auto error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(error);
    }
    ASSERT_TRUE(cuda_succeeded(cudaSetDevice(0)));
  }
};

} // namespace

TEST_F(TransportShutdownCudaTest, OpensAndDrainsAllArenas) {
  const auto first_layout = xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, 1, 1, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp16});
  const auto second_layout = xpool::transport::TransportArenaLayout::create(
      1, 0, 0, 1, 0, 1, 1, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp16});
  auto first_arena = xpool::transport::TransportArena::create(0, first_layout);
  auto second_arena = xpool::transport::TransportArena::create(0, second_layout);
  const auto views = std::array{first_arena.view(), second_arena.view()};
  auto resident = TransportResident{views};

  ASSERT_TRUE(wait_for_mailbox_status(
      first_arena, xpool::transport::MailboxStatus::Idle));
  ASSERT_TRUE(wait_for_mailbox_status(
      second_arena, xpool::transport::MailboxStatus::Idle));

  ASSERT_NO_THROW(resident.drain_async());
  ASSERT_TRUE(resident.wait());

  EXPECT_EQ(first_arena.state().shutdown, 1U);
  EXPECT_EQ(second_arena.state().shutdown, 1U);
  EXPECT_EQ(first_arena.mailbox_status(), xpool::transport::MailboxStatus::Closed);
  EXPECT_EQ(second_arena.mailbox_status(), xpool::transport::MailboxStatus::Closed);
  EXPECT_EQ(first_arena.read_generation_failure(), xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(second_arena.read_generation_failure(), xpool::abi::FfnResultCode::Ok);

  ASSERT_NO_THROW(resident.release());
  EXPECT_NO_THROW(first_arena.destroy());
  EXPECT_NO_THROW(second_arena.destroy());
}

TEST_F(TransportShutdownCudaTest, RejectsOverCapacityBeforeOpeningAnyMailbox) {
  const auto first_layout = xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, 1, 1, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp16});
  const auto second_layout = xpool::transport::TransportArenaLayout::create(
      1, 0, 0, 1, 0, 1, 1, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp16});
  auto first_arena = xpool::transport::TransportArena::create(0, first_layout);
  auto second_arena = xpool::transport::TransportArena::create(0, second_layout);
  const auto views = std::array{first_arena.view(), second_arena.view()};

  auto *view_allocation = static_cast<void *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMalloc(&view_allocation, sizeof(views))));
  auto *device_views =
      static_cast<xpool::transport::TransportArenaView *>(view_allocation);
  ASSERT_TRUE(cuda_succeeded(cudaMemcpy(device_views, views.data(), sizeof(views),
                                        cudaMemcpyHostToDevice)));

  auto *state_allocation = static_cast<void *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMalloc(
      &state_allocation, sizeof(xpool::transport::TransportResidentState))));
  auto *state = static_cast<xpool::transport::TransportResidentState *>(
      state_allocation);
  ASSERT_TRUE(cuda_succeeded(
      cudaMemset(state, 0, sizeof(xpool::transport::TransportResidentState))));
  auto stream = xpool::utils::device::OwnedCudaStream::create();

  EXPECT_THROW(
      xpool::transport::launch_transport_resident_kernel(
          device_views, std::numeric_limits<std::size_t>::max(), {}, state,
          stream.get()),
      c10::Error);
  EXPECT_EQ(first_arena.mailbox_status(),
            xpool::transport::MailboxStatus::Dormant);
  EXPECT_EQ(second_arena.mailbox_status(),
            xpool::transport::MailboxStatus::Dormant);

  EXPECT_NO_THROW(stream.destroy());
  EXPECT_TRUE(cuda_succeeded(cudaFree(state)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(device_views)));
  EXPECT_NO_THROW(first_arena.destroy());
  EXPECT_NO_THROW(second_arena.destroy());
}
