#pragma once

/// \file xpool/utils/cuda/blas.hpp
/// \brief ATen BLAS capacity discovery and exclusive-stream workspace observation.

#include <vector>

#include <xpool/utils/cuda.hpp>

namespace xpool::utils::cuda::blas {

/// Discover capacities on a fresh private stream, releasing probe owners on exit.
/// \pre The CUDA allocator is initialized; this stream has no workspace entries.
/// No GEMM runs. Call before the admission free-memory baseline.
/// \throws c10::Error when discovery or cleanup fails.
std::vector<ResourceCapacity> probe_capacity(cudaStream_t stream);

/// Borrow workspace intervals from ATen maps while capture owners remain live.
/// \throws c10::Error for a null stream, multiple handles or malformed entries.
std::vector<CaptureAllocation> snapshot_workspaces(cudaStream_t stream);

/// Synchronize and release stream-local cache owners after install or retirement.
/// \pre No further allocation or capture uses this exclusive stream.
/// The caller retains pool backing for any installed graphs borrowing it.
/// \throws c10::Error if synchronization fails or owners remain.
void release_workspaces(cudaStream_t stream);

} // namespace xpool::utils::cuda::blas
