#include <xpool/utils/cuda/blas.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

namespace xpool::utils::cuda::blas {

std::vector<CaptureAllocation> snapshot_workspaces(cudaStream_t stream) {
  TORCH_CHECK(stream != nullptr, "xpool BLAS observations require a private non-default stream");
  auto result = std::vector<CaptureAllocation>{};
  for (auto kind = 0; kind != 2; ++kind) {
    auto &storage =
        kind == 0 ? at::cuda::cublas_handle_stream_to_workspace() : at::cuda::cublaslt_handle_stream_to_workspace();
    const auto lock = std::shared_lock(storage.mutex);
    auto count = std::size_t{0};
    for (const auto &[key, value] : storage.map) {
      if (std::get<1>(key) != stream || value.second == 0) {
        continue;
      }
      TORCH_CHECK(++count == 1, "xpool serial capture requires one handle per BLAS workspace role");
      result.emplace_back(kind == 0 ? "blas.cublas" : "blas.cublaslt",
                          reinterpret_cast<std::uintptr_t>(value.first.get()), value.second);
    }
  }
  return result;
}

std::vector<ResourceCapacity> probe_capacity(cudaStream_t stream) {
  TORCH_CHECK(snapshot_workspaces(stream).empty(), "xpool BLAS probe requires a fresh private stream");
  const auto device = xpool::utils::cuda::current_device();
  const auto guard = c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(stream, device));
  auto result = std::vector<ResourceCapacity>{};
  try {
    static_cast<void>(at::cuda::getCurrentCUDABlasHandle());
    const auto lt_address = reinterpret_cast<std::uintptr_t>(at::cuda::getCUDABlasLtWorkspace());
    auto indices = std::unordered_map<std::uintptr_t, std::size_t>{};
    for (const auto &allocation : snapshot_workspaces(stream)) {
      const auto [position, inserted] = indices.emplace(allocation.address, result.size());
      if (inserted) {
        result.push_back({.key = allocation.key, .max_bytes = allocation.bytes, .alignment_bytes = 256, .aliases = {}});
      }
      auto &requirement = result[position->second];
      requirement.max_bytes = std::max(requirement.max_bytes, allocation.bytes);
      requirement.aliases.push_back(allocation.key);
    }
    if (lt_address != 0 && std::ranges::none_of(result, [](const auto &capacity) {
          return std::ranges::find(capacity.aliases, "blas.cublaslt") != capacity.aliases.end();
        })) {
      TORCH_CHECK(indices.contains(lt_address), "xpool cuBLASLt workspace has no observed backing owner");
      result[indices.at(lt_address)].aliases.emplace_back("blas.cublaslt");
    }
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  } catch (...) {
    at::cuda::clearCublasWorkspacesForStream(stream);
    throw;
  }
  at::cuda::clearCublasWorkspacesForStream(stream);
  TORCH_CHECK(snapshot_workspaces(stream).empty(), "xpool BLAS probe retained workspace owners");
  return result;
}

void release_workspaces(cudaStream_t stream) {
  TORCH_CHECK(stream != nullptr, "xpool BLAS retirement requires an exclusive stream");
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  at::cuda::clearCublasWorkspacesForStream(stream);
  TORCH_CHECK(snapshot_workspaces(stream).empty(), "xpool capture stream retained BLAS owners");
}

} // namespace xpool::utils::cuda::blas
