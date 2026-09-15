#pragma once

/// \file xpool/ffnagent/runtime.hpp
/// \brief One-time native FFN execution installation contract.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/projection.hpp>
#include <xpool/ffnagent/projection.hpp>
#include <xpool/ffnagent/runtime.cuh>
#include <xpool/macros.hpp>
#include <xpool/utils/device.hpp>

namespace xpool::ffnagent {

/// Return exact retained execution-table and lane-state allocation bytes.
/// \return Retained allocation bytes excluding graphs, workspaces, and control storage.
/// \throws c10::Error when size arithmetic overflows.
std::size_t execution_state_allocation_bytes(std::size_t local_layer_count, std::size_t local_capacity_count,
                                             std::size_t local_signature_count, std::size_t executor_lane_count);

/// Private native owner of one installed FfnAgent execution generation.
class ExecutionRuntime {
public:
  ExecutionRuntime() = default;
  ~ExecutionRuntime();

  ExecutionRuntime(const ExecutionRuntime &) = delete;
  ExecutionRuntime &operator=(const ExecutionRuntime &) = delete;
  ExecutionRuntime(ExecutionRuntime &&) = delete;
  ExecutionRuntime &operator=(ExecutionRuntime &&) = delete;

  /// Construct every lane graph and immutable lookup table.
  /// A failed attempt releases partially materialized resources but consumes
  /// this runtime; installation cannot be retried on the same owner.
  /// \pre projection passed intrinsic validation at construction.
  /// \pre fabric_projection and layout belong to the joined Fabric Runtime.
  /// \pre arena and activation_count refer to resources owned by the joined Fabric Runtime.
  void install(const ExecutionProjection &projection, xpool::fabric::ArenaView arena,
               const xpool::fabric::ArenaLayout &layout, const xpool::fabric::ArenaProjection &fabric_projection,
               std::size_t ffnagent_index, std::uint32_t *activation_count);

  /// Launch every installed Lane Graph exactly once.
  void activate();

  /// Reject premature Lane Graph completion or CUDA failure.
  void check_health() const;

  /// Return whether any activated Lane Graph remains in terminal drain.
  [[nodiscard]] bool drain_pending() const;

  /// Destroy drained lane graphs and private lookup storage.
  void finalize();

  /// Return whether a complete execution is installed.
  [[nodiscard]] bool installed() const noexcept { return installed_; }

private:
  /// One lane's owned Graph, GraphExec, stream, device tables, and workspace.
  /// Raw pointers are CUDA allocations retained until destroy() after the lane
  /// graph has drained.
  struct LaneOwner {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    xpool::utils::device::OwnedCudaStream stream;
    void *state = nullptr;
    void *schemas = nullptr;
    void *sites = nullptr;
    void *updates = nullptr;
    std::uint8_t *workspace = nullptr;
    std::size_t workspace_bytes = 0;

    LaneOwner() = default;
    ~LaneOwner();
    LaneOwner(const LaneOwner &) = delete;
    LaneOwner &operator=(const LaneOwner &) = delete;
    LaneOwner(LaneOwner &&other) noexcept;
    LaneOwner &operator=(LaneOwner &&other) noexcept;

    void destroy();
  };

  void materialize_shared_tables(const ExecutionProjection &projection,
                                 const xpool::fabric::ArenaProjection &fabric_projection, std::size_t ffnagent_index);

  void materialize_lane(const ExecutionProjection &projection, xpool::fabric::ArenaView arena,
                        const xpool::fabric::ArenaLayout &layout,
                        const xpool::fabric::ArenaProjection &fabric_projection, std::size_t ffnagent_index,
                        std::size_t executor_lane_index, int device, std::uint32_t *activation_count);

  void materialize(const ExecutionProjection &projection, xpool::fabric::ArenaView arena,
                   const xpool::fabric::ArenaLayout &layout, const xpool::fabric::ArenaProjection &fabric_projection,
                   std::size_t ffnagent_index, std::uint32_t *activation_count);

  void destroy_noexcept() noexcept;

  bool installed_ = false;
  bool active_ = false;
  bool attempted_ = false;
  void *layer_entries_ = nullptr;
  void *capacity_entries_ = nullptr;
  std::size_t layer_entry_count_ = 0;
  std::vector<LaneOwner> lanes_;
};

} // namespace xpool::ffnagent
