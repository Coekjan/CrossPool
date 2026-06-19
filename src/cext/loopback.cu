/// \file loopback.cu
/// \brief CUDA implementation of the test-oriented FFN loopback executor.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#include <loopback.hpp>

namespace {

/// Scale factor for a pairwise 45-degree rotation.
constexpr float kInvSqrt2 = 0.70710678118654752440F;

/// Rotate one pair of hidden-state columns per CUDA thread.
template <typename scalar_t>
__global__ void loopback_rotation_kernel(scalar_t *output,
                                         const scalar_t *input,
                                         std::int64_t pair_count) {
  std::int64_t pair_index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_index >= pair_count) {
    return;
  }
  std::int64_t base = pair_index * 2;
  float x = static_cast<float>(input[base]);
  float y = static_cast<float>(input[base + 1]);
  output[base] = static_cast<scalar_t>((x - y) * kInvSqrt2);
  output[base + 1] = static_cast<scalar_t>((x + y) * kInvSqrt2);
}

} // namespace

namespace xpool {

void launch_loopback_rotation(const at::Tensor &output,
                              const at::Tensor &input) {
  TORCH_CHECK(input.is_cuda(), "xpool loopback input must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "xpool loopback output must be a CUDA tensor");
  TORCH_CHECK(input.device() == output.device(),
              "xpool loopback input and output must be on the same device");
  TORCH_CHECK(input.sizes() == output.sizes(),
              "xpool loopback input and output must have identical shapes");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "xpool loopback input and output must have identical dtypes");
  TORCH_CHECK(input.is_contiguous(), "xpool loopback input must be contiguous");
  TORCH_CHECK(output.is_contiguous(),
              "xpool loopback output must be contiguous");
  TORCH_CHECK(input.dim() == 2, "xpool loopback input must be 2D");
  TORCH_CHECK(input.size(1) % 2 == 0,
              "xpool loopback hidden size must be even");

  std::int64_t pair_count = input.numel() / 2;
  if (pair_count == 0) {
    return;
  }

  constexpr int kThreadsPerBlock = 256;
  std::int64_t block_count =
      (pair_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  TORCH_CHECK(block_count <= std::numeric_limits<int>::max(),
              "xpool loopback tensor is too large for a single CUDA grid");
  int blocks = static_cast<int>(block_count);
  c10::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "xpool_loopback_rotation",
      [&] {
        loopback_rotation_kernel<scalar_t>
            <<<blocks, kThreadsPerBlock, 0, stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                pair_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xpool
