#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <xpool/abi.hpp>
#include <xpool/atnagent.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/instance.hpp>
#include <xpool/transport.hpp>
#include <xpool/utils/arith.hpp>

namespace {

using RuntimeRole = xpool::abi::RuntimeRole;

struct RuntimeState {
  std::optional<RuntimeRole::Type> role;
};

std::mutex g_runtime_state_mutex;
RuntimeState g_runtime_state;

void require_role(RuntimeRole::Type expected, const char *op_name) {
  std::lock_guard<std::mutex> lock(g_runtime_state_mutex);
  RuntimeRole::expect(g_runtime_state.role, expected, op_name);
}

void init(std::int64_t cuda_device, std::int64_t role,
          std::int64_t debug_options) {
  TORCH_CHECK(cuda_device >= 0,
              "xpool init requires a non-negative CUDA device");
  RuntimeRole::Type parsed_role = RuntimeRole::parse(role);
  std::lock_guard<std::mutex> lock(g_runtime_state_mutex);
  if (!g_runtime_state.role.has_value()) {
    g_runtime_state.role = parsed_role;
  }
  RuntimeRole::expect(g_runtime_state.role, parsed_role, "init");
  xpool::debug::init(cuda_device, debug_options);
}

} // namespace

// Register the xpool Torch operator schemas.
TORCH_LIBRARY(xpool, m) {
  {
    m.def("abi_version() -> int");
    const auto f = []() {
      return static_cast<std::int64_t>(xpool::abi::kAbiVersion);
    };
    m.impl("abi_version", f);
  }

  {
    m.def("init(int cuda_device, int role, int debug_options) -> ()");
    m.impl("init", init);
  }

  {
    m.def("atnagent.create_transport_arena("
          "int cuda_device, "
          "int max_tokens, "
          "int hidden_size, "
          "int element_size_bytes, "
          "int atn_dp_size"
          ") -> str");
    const auto f = [](std::int64_t cuda_device, std::int64_t max_tokens,
                      std::int64_t hidden_size, std::int64_t element_size_bytes,
                      std::int64_t atn_dp_size) {
      require_role(RuntimeRole::kAtnagent, "atnagent.create_transport_arena");
      return xpool::atnagent::create_transport_arena(
          cuda_device, max_tokens, hidden_size, element_size_bytes,
          atn_dp_size);
    };
    m.impl("atnagent.create_transport_arena", f);
  }

  {
    m.def("atnagent.launch_transport_kernel(str handle) -> ()");
    const auto f = [](const xpool::transport::TransportArenaHandleHex &handle) {
      require_role(RuntimeRole::kAtnagent, "atnagent.launch_transport_kernel");
      xpool::atnagent::launch_transport_kernel(handle);
    };
    m.impl("atnagent.launch_transport_kernel", f);
  }

  {
    m.def("atnagent.destroy_transport_arena(str handle) -> "
          "(int sequence, int dropped, int[][] records)");
    const auto f = [](const xpool::transport::TransportArenaHandleHex &handle) {
      require_role(RuntimeRole::kAtnagent, "atnagent.destroy_transport_arena");
      const xpool::abi::TransportTraceSnapshot snapshot =
          xpool::atnagent::destroy_transport_arena(handle);
      std::vector<std::vector<std::int64_t>> records;
      records.reserve(snapshot.records.size());
      for (const xpool::abi::TransportTraceRecord &record : snapshot.records) {
        records.push_back({
            static_cast<std::int64_t>(record.trace_id),
            static_cast<std::int64_t>(record.slot),
            static_cast<std::int64_t>(record.num_tokens),
            static_cast<std::int64_t>(record.request_begin),
            static_cast<std::int64_t>(record.slot_claimed),
            static_cast<std::int64_t>(record.input_staged),
            static_cast<std::int64_t>(record.request_published),
            static_cast<std::int64_t>(record.atnagent_dequeued),
            static_cast<std::int64_t>(record.descriptor_granted),
            static_cast<std::int64_t>(record.executor_begin),
            static_cast<std::int64_t>(record.executor_end),
            static_cast<std::int64_t>(record.result_published),
            static_cast<std::int64_t>(record.result_observed),
            static_cast<std::int64_t>(record.output_copied),
            static_cast<std::int64_t>(record.slot_recycled),
        });
      }
      return std::make_tuple(static_cast<std::int64_t>(snapshot.sequence),
                             static_cast<std::int64_t>(snapshot.dropped),
                             std::move(records));
    };
    m.impl("atnagent.destroy_transport_arena", f);
  }

  {
    m.def("instance.attach_transport_arena("
          "int instance_index, "
          "int rank, "
          "str handle"
          ") -> ()");
    const auto f = [](std::int64_t instance_index, std::int64_t rank,
                      const xpool::transport::TransportArenaHandleHex &handle) {
      require_role(RuntimeRole::kInstance, "instance.attach_transport_arena");
      xpool::instance::attach_transport_arena(instance_index, rank, handle);
    };
    m.impl("instance.attach_transport_arena", f);
  }

  {
    m.def(
        "instance.detach_transport_arena(int instance_index, int rank) -> ()");
    const auto f = [](std::int64_t instance_index, std::int64_t rank) {
      require_role(RuntimeRole::kInstance, "instance.detach_transport_arena");
      xpool::instance::detach_transport_arena(instance_index, rank);
    };
    m.impl("instance.detach_transport_arena", f);
  }

  {
    m.def("instance.transport_error_snapshot(int instance_index, int rank) -> "
          "int");
    const auto f = [](std::int64_t instance_index, std::int64_t rank) {
      require_role(RuntimeRole::kInstance, "instance.transport_error_snapshot");
      return static_cast<std::int64_t>(
          xpool::instance::transport_error_snapshot(instance_index, rank));
    };
    m.impl("instance.transport_error_snapshot", f);
  }

  {
    m.def("instance.ffn_shim("
          "Tensor hidden_states, "
          "Tensor? global_num_tokens_gpu, "
          "int instance_index, "
          "int rank, "
          "int layer_id, "
          "int forward_mode, "
          "int collective_policy, "
          "int dp_padding_mode, "
          "int global_dp_buffer_len, "
          "int atn_tp_rank, "
          "int atn_tp_size, "
          "int atn_dp_rank, "
          "int atn_dp_size"
          ") -> Tensor");
    const auto f =
        [](const at::Tensor &hidden_states,
           const std::optional<at::Tensor> &global_num_tokens_gpu,
           std::int64_t instance_index, std::int64_t rank,
           std::int64_t layer_id, std::int64_t forward_mode,
           std::int64_t collective_policy, std::int64_t dp_padding_mode,
           std::int64_t global_dp_buffer_len, std::int64_t atn_tp_rank,
           std::int64_t atn_tp_size, std::int64_t atn_dp_rank,
           std::int64_t atn_dp_size) -> at::Tensor {
      require_role(RuntimeRole::kInstance, "instance.ffn_shim");
      const xpool::abi::FfnRequestMetadata request_metadata{
          XPOOL_CHECKED_U32(instance_index),
          XPOOL_CHECKED_U32(layer_id),
          XPOOL_CHECKED_U32(forward_mode),
          XPOOL_CHECKED_U32(collective_policy),
          XPOOL_CHECKED_U32(dp_padding_mode),
          XPOOL_CHECKED_U32(atn_tp_rank),
          XPOOL_CHECKED_U32(atn_tp_size),
          XPOOL_CHECKED_U32(atn_dp_rank),
          XPOOL_CHECKED_U32(atn_dp_size),
          XPOOL_CHECKED_U32(global_dp_buffer_len),
      };
      return xpool::instance::ffn_shim(hidden_states, request_metadata, rank,
                                       global_num_tokens_gpu);
    };
    m.impl("instance.ffn_shim", f);
  }
}
