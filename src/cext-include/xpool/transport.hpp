#pragma once

/// \file xpool/transport.hpp
/// \brief Internal CUDA transport kernel launch declarations.

#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>

#include <xpool/abi.hpp>
#include <xpool/utils/hex.hpp>
#include <xpool/utils/queue.hpp>
#ifdef __CUDACC__
#include <xpool/utils/queue.cuh>
#endif

namespace xpool::transport {

/// Lowercase hex CUDA IPC handle exchanged through native control-plane APIs.
using TransportArenaHandleHex = std::string;

/// Byte alignment used for transport arena regions and per-slot buffers.
inline constexpr std::int64_t kTransportArenaAlignment = 256;
/// Default number of reusable request slots in each transport arena.
///
/// The current protocol supports one in-flight request per arena. Future
/// microbatching or arena-local concurrency can use a different slot count
/// without exposing slot-count policy through the Python control-plane API.
inline constexpr std::int64_t kTransportArenaSlotCountDefault = 1;
/// Magic value stored at the beginning of every transport arena layout header.
inline constexpr std::uint64_t kTransportArenaLayoutMagic =
    0x3141544c4f4f5058ULL;

/// CUDA thread count for the instance-side one-shot transport request kernel.
inline constexpr int kRequestThreadsPerBlock = 256;
/// CUDA thread count for the devagent-side resident transport kernel.
inline constexpr int kDevagentThreadsPerBlock = 32;
/// Device clock budget used by transport queue waits before trapping.
inline constexpr unsigned long long kDeviceSpinTimeoutClocks =
    xpool::utils::queue::kDefaultRingQueueSpinTimeoutClocks;
/// Number of recent requests retained by an observed transport arena.
inline constexpr std::int64_t kTransportTraceCapacity = 8192;

/// CUDA IPC handle that identifies one devagent-owned transport arena.
struct TransportArenaHandle {
  /// Raw CUDA IPC memory handle returned by cudaIpcGetMemHandle.
  cudaIpcMemHandle_t value;

  /// Parse a lowercase hex CUDA IPC handle.
  /// \param hex Lowercase hex handle exchanged through Python/daemon APIs.
  /// \return Strongly typed CUDA IPC arena handle.
  /// \throws c10::Error if the string is not a CUDA IPC handle.
  static TransportArenaHandle from_hex(const TransportArenaHandleHex &hex) {
    const auto bytes = xpool::utils::hex::decode_lowercase(hex);
    TORCH_CHECK(bytes.size() == sizeof(cudaIpcMemHandle_t),
                "xpool transport arena handle has unexpected byte length");
    TransportArenaHandle handle{};
    std::memcpy(&handle.value, bytes.data(), sizeof(handle.value));
    return handle;
  }

  /// Return the lowercase hex representation used by Python and daemon APIs.
  /// \return Lowercase hex CUDA IPC handle.
  TransportArenaHandleHex hex() const {
    return xpool::utils::hex::encode_lowercase(&value, sizeof(value));
  }

  /// Return whether two handles refer to the same opaque CUDA IPC value.
  /// \param other Handle to compare with this handle.
  /// \return True when the raw CUDA IPC handle bytes match.
  bool operator==(const TransportArenaHandle &other) const noexcept {
    return std::memcmp(&value, &other.value, sizeof(value)) == 0;
  }
};

/// Host-visible byte layout of one CUDA IPC transport arena.
struct TransportArenaLayout {
  /// Fixed magic value that identifies an xpool transport arena header.
  std::uint64_t magic;
  /// Total byte size of the CUDA IPC arena allocation.
  std::int64_t arena_bytes;
  /// Number of reusable descriptor/input/output slots in the arena.
  std::int64_t slot_count;
  /// Byte stride between per-slot input buffers and output buffers.
  std::int64_t slot_stride_bytes;
  /// Queue whose cells carry slots available for instance producers.
  xpool::utils::queue::RingQueueLayout free_queue;
  /// Queue whose cells carry slots published for devagent consumers.
  xpool::utils::queue::RingQueueLayout used_queue;
  /// Byte offset of the arena-wide shutdown flag.
  std::int64_t shutdown_offset;
  /// Byte offset of the first sticky fatal executor error code.
  std::int64_t error_code_offset;
  /// Byte offset of the FfnRequestDescriptor array.
  std::int64_t request_base_offset;
  /// Byte offset of the FfnResultDescriptor array.
  std::int64_t result_base_offset;
  /// Byte offset of the per-slot input staging buffer array.
  std::int64_t input_base_offset;
  /// Byte offset of the per-slot output staging buffer array.
  std::int64_t output_base_offset;
  /// Byte offset of the per-slot DP token-count array.
  std::int64_t dp_token_counts_base_offset;
  /// Byte stride between per-slot DP token-count buffers.
  std::int64_t dp_token_counts_stride_bytes;
  /// Maximum token rows accepted by each request slot.
  std::int64_t max_tokens;
  /// Hidden-state width accepted by each request slot.
  std::int64_t hidden_size;
  /// Attention data-parallel world size used to allocate DP token-count
  /// buffers.
  std::int64_t atn_dp_size;
  /// Byte offset of the observer sequence counter, or zero when disabled.
  std::int64_t trace_sequence_offset;
  /// Byte offset of the observer overwrite counter, or zero when disabled.
  std::int64_t trace_dropped_offset;
  /// Byte offset of the TransportTraceRecord ring, or zero when disabled.
  std::int64_t trace_records_offset;
  /// Number of records retained by the observer ring; zero when disabled.
  std::int64_t trace_capacity;
  /// Return whether two arena layouts describe the same byte geometry.
  /// \return True when every layout field matches exactly.
  bool operator==(const TransportArenaLayout &) const = default;

  /// Build the arena geometry for one transport arena.
  /// \param slot_count Number of reusable request slots.
  /// \param max_tokens Maximum token rows supported by one slot.
  /// \param hidden_size Hidden-state width supported by one slot.
  /// \param element_size_bytes Bytes per hidden-state element.
  /// \param atn_dp_size Attention data-parallel world size for DP token counts.
  /// \throws c10::Error if any parameter is invalid or geometry overflows.
  TransportArenaLayout(std::int64_t slot_count, std::int64_t max_tokens,
                       std::int64_t hidden_size,
                       std::int64_t element_size_bytes,
                       std::int64_t atn_dp_size);
  /// Validate one FFN request against this arena geometry.
  /// \param tensor_metadata Hidden-state tensor facts parsed from the request
  /// tensor.
  /// \param request_metadata FFN request metadata parsed from host arguments.
  /// \throws c10::Error if the request cannot fit in this arena.
  void validate(const xpool::abi::FfnTensorMetadata &tensor_metadata,
                const xpool::abi::FfnRequestMetadata &request_metadata) const;
};

/// Process-local view over one CUDA IPC transport arena mapping.
struct TransportArena {
  /// Kind of release required for the process-local base pointer.
  enum class ReleaseKind {
    /// Arena allocation is owned by this process and must be cudaFree'd.
    kOwnedAllocation,
    /// Arena mapping was opened by CUDA IPC and must be closed.
    kIpcMapping,
  };

  /// Process-local byte-addressable base pointer for the CUDA arena.
  std::uint8_t *base = nullptr;
  /// Release operation required by destroy(); ignored when base is null.
  ReleaseKind release_kind = ReleaseKind::kOwnedAllocation;

  /// Allocate and fully initialize a devagent-owned arena.
  /// \param cuda_device CUDA device index that owns the allocation.
  /// \param layout Host-computed arena layout written into the arena header.
  /// \return Process-local owner view for the allocated arena.
  static TransportArena create(std::int64_t cuda_device,
                               const TransportArenaLayout &layout);
  /// Open an instance-side CUDA IPC mapping for an existing arena handle.
  /// \param handle CUDA IPC memory handle exported by the devagent process.
  /// \return Process-local IPC mapping view for the existing arena.
  static TransportArena from_handle(const TransportArenaHandle &handle) {
    std::uint8_t *arena = nullptr;
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&arena),
                                        handle.value,
                                        cudaIpcMemLazyEnablePeerAccess));
    return TransportArena{arena, ReleaseKind::kIpcMapping};
  }
  /// Open an instance-side CUDA IPC mapping for an existing hex arena handle.
  /// \param handle Lowercase hex CUDA IPC handle from the daemon.
  /// \return Process-local IPC mapping view for the existing arena.
  static TransportArena from_handle_hex(const TransportArenaHandleHex &handle) {
    return from_handle(TransportArenaHandle::from_hex(handle));
  }
  /// Release the arena pointer according to release_kind.
  void destroy();
  /// Export a CUDA IPC handle for a devagent-owned arena.
  /// \return CUDA IPC memory handle for the owned arena allocation.
  TransportArenaHandle handle() const {
    TORCH_CHECK(base != nullptr,
                "xpool cannot export a CUDA IPC handle for an empty arena");
    TORCH_CHECK(release_kind == ReleaseKind::kOwnedAllocation,
                "xpool can only export CUDA IPC handles for owned arenas");
    TransportArenaHandle handle{};
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle.value, base));
    return handle;
  }
  /// Copy and identify the arena layout header from device to host.
  /// \return Host copy of the arena layout header.
  TransportArenaLayout layout() const {
    TORCH_CHECK(base != nullptr,
                "xpool cannot read layout from an empty transport arena");
    std::array<std::byte, sizeof(TransportArenaLayout)> bytes{};
    C10_CUDA_CHECK(
        cudaMemcpy(bytes.data(), base, bytes.size(), cudaMemcpyDeviceToHost));
    TransportArenaLayout layout = std::bit_cast<TransportArenaLayout>(bytes);
    TORCH_CHECK(layout.magic == kTransportArenaLayoutMagic,
                "xpool transport arena header magic does not match");
    return layout;
  }
  /// Copy observer records and counters into a structured host snapshot.
  /// \return ABI snapshot with empty records when observation is disabled.
  xpool::abi::TransportTraceSnapshot trace_snapshot() const;
  /// Copy the sticky fatal executor error code to host memory.
  /// \return FfnResultErrorCode value, or kOk when no error was recorded.
  std::uint32_t error_code_snapshot() const;
  /// Ask a resident devagent kernel to drain and stop.
  /// \param stream CUDA stream used to publish the shutdown flag.
  void request_shutdown(cudaStream_t stream) const {
    TORCH_CHECK(base != nullptr,
                "xpool cannot request shutdown on an empty transport arena");
    const TransportArenaLayout host_layout = layout();
    auto value = static_cast<std::uint32_t>(1);
    C10_CUDA_CHECK(cudaMemcpyAsync(base + host_layout.shutdown_offset, &value,
                                   sizeof(value), cudaMemcpyHostToDevice,
                                   stream));
  }

#ifdef __CUDACC__
  /// Return the arena-wide shutdown flag.
  /// \return Reference to the IPC-visible shutdown flag.
  __device__ std::uint32_t &shutdown() const;
  /// Return the arena-wide sticky executor error code.
  /// \return Reference to the IPC-visible FfnResultErrorCode value.
  __device__ std::uint32_t &error_code() const;
  /// Return one request descriptor slot.
  /// \param slot Slot id previously acquired from the free or used queue.
  /// \return Reference to the request descriptor for slot.
  __device__ xpool::abi::FfnRequestDescriptor &
  request(std::uint32_t slot) const;
  /// Return one result descriptor slot.
  /// \param slot Slot id previously acquired from the free or used queue.
  /// \return Reference to the result descriptor for slot.
  __device__ xpool::abi::FfnResultDescriptor &result(std::uint32_t slot) const;
  /// Reserve and initialize one transport observer record.
  /// \param num_tokens Request token rows recorded for analysis.
  /// \return Record pointer, or null when observation is disabled.
  __device__ xpool::abi::TransportTraceRecord *
  begin_trace(std::uint32_t num_tokens) const;
  /// Find an observer record by monotonic trace id.
  /// \param trace_id Trace id carried by the request descriptor.
  /// \return Matching record pointer, or null when unavailable or overwritten.
  __device__ xpool::abi::TransportTraceRecord *
  trace(std::uint64_t trace_id) const;
  /// Return one typed input payload slot.
  /// \tparam value_t Element type stored in the staged input buffer.
  /// \param slot Slot id reserved by the instance request kernel.
  /// \return Device pointer to the start of the slot's input buffer.
  template <typename value_t>
  __device__ value_t *input_buffer(std::uint32_t slot) const;
  /// Return one typed output payload slot.
  /// \tparam value_t Element type stored in the staged output buffer.
  /// \param slot Slot id reserved by the instance request kernel.
  /// \return Device pointer to the start of the slot's output buffer.
  template <typename value_t>
  __device__ value_t *output_buffer(std::uint32_t slot) const;
  /// Return one staged DP token-count payload slot.
  /// \param slot Slot id reserved by the instance request kernel.
  /// \return Device pointer to the slot's uint32 DP token-count buffer.
  __device__ std::uint32_t *dp_token_counts_buffer(std::uint32_t slot) const;
  /// Return the reusable free-slot queue.
  /// \return Device queue view whose cells carry idle slot ids.
  __device__ xpool::utils::queue::RingQueue<std::uint32_t> free_queue() const;
  /// Return the published used-slot queue.
  /// \return Device queue view whose cells carry published request slot ids.
  __device__ xpool::utils::queue::RingQueue<std::uint32_t> used_queue() const;
  /// Return descriptor offsets for one FFN request slot.
  /// \param slot Slot id reserved by the instance request kernel.
  /// \param has_dp_token_counts Whether this request staged DP token counts.
  /// \return Arena offsets stamped into the request descriptor.
  __device__ xpool::abi::FfnArenaOffsets
  ffn_offsets(std::uint32_t slot, bool has_dp_token_counts) const;
  /// Return the input payload referenced by a published FFN request.
  /// \tparam value_t Element type selected from request tensor metadata.
  /// \param request Published request descriptor with arena offsets.
  /// \return Device pointer to the request input payload.
  template <typename value_t>
  __device__ const value_t *
  request_input(const xpool::abi::FfnRequestDescriptor &request) const;
  /// Return the output payload referenced by a published FFN request.
  /// \tparam value_t Element type selected from request tensor metadata.
  /// \param request Published request descriptor with arena offsets.
  /// \return Device pointer to the request output payload.
  template <typename value_t>
  __device__ value_t *
  request_output(const xpool::abi::FfnRequestDescriptor &request) const;

private:
  __device__ const TransportArenaLayout &arena_layout() const;
  __device__ std::uint64_t slot_count() const;
  __device__ void check_range(std::uint64_t offset, std::uint64_t bytes) const;
  __device__ void check_slot(std::uint32_t slot) const;
  template <typename value_t>
  __device__ value_t *pointer_at(std::uint64_t offset,
                                 std::uint64_t bytes = sizeof(value_t)) const;
  __device__ std::uint64_t slot_offset(std::int64_t base_offset,
                                       std::uint32_t slot,
                                       std::int64_t stride_bytes,
                                       std::uint64_t bytes) const;
  __device__ std::uint64_t slot_offset(std::int64_t base_offset,
                                       std::uint32_t slot,
                                       std::int64_t stride_bytes) const;
  template <typename value_t>
  __device__ value_t &array_at(std::int64_t base_offset,
                               std::uint32_t slot) const;
  template <typename value_t>
  __device__ value_t *slot_buffer_at(std::int64_t base_offset,
                                     std::uint32_t slot,
                                     std::int64_t stride_bytes) const;
  template <typename value_t>
  __device__ static std::uint64_t
  tensor_bytes(const xpool::abi::FfnTensorMetadata &metadata);
#endif
};

static_assert(std::is_standard_layout_v<TransportArenaHandle>);
static_assert(std::is_trivially_copyable_v<TransportArenaHandle>);
static_assert(std::is_standard_layout_v<TransportArenaLayout>);
static_assert(std::is_trivially_copyable_v<TransportArenaLayout>);

/// Host-side request object for launching one instance-to-devagent FFN request.
///
/// This object is not the device-side FfnRequestDescriptor. It only carries
/// host-side launch arguments used to stage and publish one descriptor.
struct TransportRequest {
  /// CUDA stream that owns staging, request publication, and completion wait.
  cudaStream_t stream;
  /// Process-local base pointer for the mapped CUDA IPC arena.
  TransportArena arena;
  /// Contiguous CUDA hidden-state input tensor.
  const at::Tensor &hidden_states;
  /// Optional per-DP-rank token counts for this request.
  const std::optional<at::Tensor> &global_num_tokens_gpu;
  /// CUDA output tensor that receives the completed hidden states.
  const at::Tensor &output;
  /// Request metadata stamped into the device-visible FFN request.
  const xpool::abi::FfnRequestMetadata &request_metadata;
  /// Tensor facts stamped into the device-visible FFN request.
  const xpool::abi::FfnTensorMetadata &tensor_metadata;
};

/// Launch the resident devagent-side transport kernel for one arena.
/// \param arena Process-local arena view owned by the devagent process.
/// \param stream CUDA stream that owns the resident kernel launch.
void launch_devagent_transport_kernel(TransportArena arena,
                                      cudaStream_t stream);
/// Launch one instance-side transport kernel for a single FFN request.
/// \param request Host launch arguments and descriptor metadata.
void launch_instance_transport_kernel(const TransportRequest &request);

} // namespace xpool::transport
