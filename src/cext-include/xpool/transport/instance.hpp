#pragma once

/// \file xpool/transport/instance.hpp
/// \brief Single CUDA IPC transport attachment owned by one Instance process.

#include <ATen/core/TensorBody.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

#include <xpool/transport/arena.hpp>

namespace xpool::transport {

/// Host-side launch arguments for one Instance transport request.
struct TransportRequest {
  /// CUDA stream that owns staging, publication, and completion waiting.
  cudaStream_t stream;
  /// Process-local view over the attached CUDA IPC arena.
  TransportArenaView arena;
  /// Contiguous CUDA hidden-state input tensor.
  const at::Tensor &hidden_states;
  /// Optional per-DP-rank token counts for this request.
  const std::optional<at::Tensor> &global_num_tokens_gpu;
  /// CUDA output tensor that receives completed hidden states.
  const at::Tensor &output;
  /// Request identity and parallelism metadata.
  const FfnRequestMetadata &request_metadata;
};

/// Launch one Instance-side transport kernel.
/// \param request Validated host launch arguments and request metadata.
/// \throws c10::Error if request validation or kernel launch fails.
void launch_request_kernel(const TransportRequest &request);

/// Process-lifetime owner of one Instance-side transport attachment.
///
/// The attachment must be explicitly detached before process teardown. Static
/// destruction terminates empty host state and is not a recovery path for a
/// live CUDA IPC mapping.
class InstanceTransportRuntime {
public:
  /// Return the sole process-lifetime Instance transport runtime.
  /// \return Instance transport lifecycle owner for this process.
  static InstanceTransportRuntime &singleton() {
    static InstanceTransportRuntime runtime;
    return runtime;
  }

  InstanceTransportRuntime(const InstanceTransportRuntime &) = delete;
  InstanceTransportRuntime &operator=(const InstanceTransportRuntime &) = delete;
  InstanceTransportRuntime(InstanceTransportRuntime &&) = delete;
  InstanceTransportRuntime &operator=(InstanceTransportRuntime &&) = delete;

  /// Open the process's sole CUDA IPC transport attachment.
  /// \param instance_index Config-order instance identity.
  /// \param rank Rank-local process index within the instance.
  /// \param handle Daemon-brokered CUDA IPC arena handle.
  void attach_arena(std::size_t instance_index, std::size_t rank, const TransportArenaHandle &handle);

  /// Synchronize, close, and clear the process's attachment.
  /// \post No attachment remains after successful cleanup.
  void detach_arena();

  /// Read the sticky first non-shutdown failure from the attachment.
  /// \return Canonical Fabric failure, or Ok when none was recorded.
  xpool::abi::FfnResultCode read_generation_failure() const;

  /// Submit one FFN request through the process's attachment.
  /// \param hidden_states Contiguous CUDA tensor shaped [tokens, hidden].
  /// \param global_num_tokens_gpu Optional contiguous CUDA int32 or int64 DP token-count tensor.
  /// \param request_metadata Validated request and instance identity.
  /// \return CUDA output tensor ordered after transport completion.
  at::Tensor submit(const at::Tensor &hidden_states, const std::optional<at::Tensor> &global_num_tokens_gpu,
                    const xpool::transport::FfnRequestMetadata &request_metadata);

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
    void attach(std::size_t instance_index, std::size_t rank, const TransportArenaHandle &handle);

    /// Synchronize and close this process's transport attachment.
    void detach();

    /// Return whether the live attachment has the supplied identity.
    bool matches(std::size_t instance_index, std::size_t rank, const TransportArenaHandle &handle) const {
      const auto &layout = arena_.layout();
      return layout.instance_index == instance_index && layout.instance_rank == rank && handle_ == handle;
    }

    /// Read the sticky canonical generation failure from the attached arena.
    xpool::abi::FfnResultCode read_generation_failure() const;

    /// Submit one FFN request through the attached transport arena.
    at::Tensor submit(const at::Tensor &hidden_states, const std::optional<at::Tensor> &global_num_tokens_gpu,
                      const xpool::transport::FfnRequestMetadata &request_metadata) const;

  private:
    TransportArenaHandle handle_;
    TransportArena arena_;
  };

  InstanceTransportRuntime() = default;
  ~InstanceTransportRuntime() = default;

  mutable std::mutex mutex_;
  Attachment attachment_;
};

} // namespace xpool::transport
