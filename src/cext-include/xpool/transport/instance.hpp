#pragma once

/// \file xpool/transport/instance.hpp
/// \brief Single CUDA IPC transport attachment owned by one Instance-rank process.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

#include <ATen/core/TensorBody.h>

#include <xpool/transport/arena.hpp>

namespace xpool::transport {

/// Host-side launch arguments for one Instance transport request.
struct Request {
  /// CUDA stream that owns staging, publication, and completion waiting.
  cudaStream_t stream;
  /// Process-local view over the attached CUDA IPC arena.
  ArenaView arena;
  /// Contiguous CUDA hidden-state input tensor.
  const at::Tensor &hidden_states;
  /// Optional physical row spans for every DP rank in this request.
  const std::optional<at::Tensor> &dp_rank_payload_rows;
  /// CUDA output tensor that receives completed hidden states.
  const at::Tensor &output;
  /// Request identity and parallelism metadata.
  const RequestMetadata &request_metadata;
};

/// Launch one Instance-side transport kernel.
/// \throws c10::Error if request validation or kernel launch fails.
void launch_request_kernel(const Request &request);

/// Process-lifetime owner of one Instance-side transport attachment.
///
/// Explicit detach owns controlled in-process cleanup. Process loss instead
/// relies on CUDA context teardown and Generation fail-stop; static destruction
/// is not a recovery path for a live CUDA IPC mapping. Public operations require
/// the process role and attachment lifecycle they name; identity, CUDA IPC,
/// kernel, and protocol failures surface as c10::Error.
class InstanceRankRuntime {
public:
  /// Return the sole process-lifetime Instance-rank Transport runtime.
  static InstanceRankRuntime &singleton() {
    static InstanceRankRuntime runtime;
    return runtime;
  }

  InstanceRankRuntime(const InstanceRankRuntime &) = delete;
  InstanceRankRuntime &operator=(const InstanceRankRuntime &) = delete;
  InstanceRankRuntime(InstanceRankRuntime &&) = delete;
  InstanceRankRuntime &operator=(InstanceRankRuntime &&) = delete;

  /// Open the process's sole CUDA IPC transport attachment.
  void attach_arena(std::size_t instance_index, std::size_t rank, const ArenaHandle &handle);

  /// Synchronize, close, and clear the process's attachment.
  /// \post No attachment remains after successful cleanup.
  void detach_arena();

  /// Read the sticky first non-shutdown failure from the attachment.
  xpool::ffn::ResultCode read_generation_failure() const;

  /// Submit one FFN request through the process's attachment.
  void submit(const at::Tensor &hidden_states, const std::optional<at::Tensor> &dp_rank_payload_rows,
              const at::Tensor &output, const xpool::transport::RequestMetadata &request_metadata);

private:
  class Attachment {
  public:
    /// Construct an empty transport attachment.
    Attachment() = default;

    Attachment(const Attachment &) = delete;
    Attachment &operator=(const Attachment &) = delete;
    Attachment(Attachment &&) = delete;
    Attachment &operator=(Attachment &&) = delete;

    /// Return whether this attachment owns an open CUDA IPC mapping.
    explicit operator bool() const noexcept { return static_cast<bool>(arena_); }

    /// Open or validate this process's transport attachment.
    void attach(std::size_t instance_index, std::size_t rank, const ArenaHandle &handle);

    /// Synchronize and close this process's transport attachment.
    void detach();

    /// Return whether the live attachment has the supplied identity.
    bool matches(std::size_t instance_index, std::size_t rank, const ArenaHandle &handle) const {
      const auto &layout = arena_.layout();
      return layout.instance_index == instance_index && layout.instance_rank == rank && handle_ == handle;
    }

    /// Read the sticky canonical generation failure from the attached arena.
    xpool::ffn::ResultCode read_generation_failure() const;

    /// Submit one FFN request through the attached transport arena.
    void submit(const at::Tensor &hidden_states, const std::optional<at::Tensor> &dp_rank_payload_rows,
                const at::Tensor &output, const xpool::transport::RequestMetadata &request_metadata) const;

  private:
    ArenaHandle handle_;
    Arena arena_;
  };

  InstanceRankRuntime() = default;
  ~InstanceRankRuntime() = default;

  mutable std::mutex mutex_;
  Attachment attachment_;
};

} // namespace xpool::transport
