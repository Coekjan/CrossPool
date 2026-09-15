#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/stl.h>
#include <torch/python.h>

#include "bindings.hpp"
#include <xpool/ffn.hpp>
#include <xpool/runtime.hpp>
#include <xpool/transport/atnagent.hpp>
#include <xpool/transport/instance.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_transport(py::module_ &module) {
  auto transport = module.def_submodule("transport", "Native CUDA IPC transport control and lifecycle functions.");
  transport.attr("ARENA_HANDLE_HEX_LENGTH") = xpool::transport::ArenaHandle::encoded_size;
  transport.def(
      "create_arena",
      [](std::size_t instance_index, std::size_t instance_rank, std::size_t payload_row_capacity,
         std::size_t hidden_size, const py::object &dtype, std::size_t atn_tp_rank, std::size_t atn_tp_size,
         std::size_t atn_dp_rank, std::size_t atn_dp_size) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.create_arena");
        const auto payload_dtype = torch::python::detail::py_object_to_dtype(dtype);
        TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(payload_dtype),
                    "xpool Transport payload dtype must be torch.float16 or torch.bfloat16");
        return xpool::transport::AtnAgentRuntime::singleton()
            .create_arena(xpool::RuntimeState::singleton().cuda_device("xpool.native.transport.create_arena"),
                          instance_index, instance_rank, payload_row_capacity, hidden_size, payload_dtype, atn_tp_rank,
                          atn_tp_size, atn_dp_rank, atn_dp_size)
            .encode();
      },
      py::arg("instance_index"), py::arg("instance_rank"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
      py::arg("dtype"), py::arg("atn_tp_rank"), py::arg("atn_tp_size"), py::arg("atn_dp_rank"), py::arg("atn_dp_size"),
      "Create one AtnAgent-owned CUDA IPC Transport arena.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "activate",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent, "xpool.native.transport.activate");
        xpool::transport::AtnAgentRuntime::singleton().activate();
      },
      "Launch resident Transport kernels for every created arena.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "check_health",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.check_health");
        xpool::transport::AtnAgentRuntime::singleton().check_health();
      },
      "Raise when an active Transport resident has failed.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "drain_async",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.drain_async");
        xpool::transport::AtnAgentRuntime::singleton().drain_async();
      },
      "Begin asynchronous drain for every active Transport resident.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "drain_pending",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.drain_pending");
        return xpool::transport::AtnAgentRuntime::singleton().drain_pending();
      },
      "Return whether any Transport resident drain remains pending.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "destroy_arenas",
      [](const std::vector<std::string> &handles) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.destroy_arenas");
        auto decoded = std::vector<xpool::transport::ArenaHandle>{};
        decoded.reserve(handles.size());
        for (const auto &handle : handles) {
          decoded.push_back(xpool::transport::ArenaHandle::decode(handle));
        }
        xpool::transport::AtnAgentRuntime::singleton().destroy_arenas(decoded);
      },
      py::arg("handles"), "Destroy the specified drained AtnAgent-owned arenas.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "attach_arena",
      [](std::size_t instance_index, std::size_t rank, const std::string &handle) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.attach_arena");
        xpool::transport::InstanceRankRuntime::singleton().attach_arena(instance_index, rank,
                                                                        xpool::transport::ArenaHandle::decode(handle));
      },
      py::arg("instance_index"), py::arg("rank"), py::arg("handle"),
      "Attach one Instance-rank process to its rank-local CUDA IPC arena.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "detach_arena",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.detach_arena");
        xpool::transport::InstanceRankRuntime::singleton().detach_arena();
      },
      "Detach the Instance-rank process from its current CUDA IPC arena.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "read_generation_failure",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.read_generation_failure");
        return xpool::transport::InstanceRankRuntime::singleton().read_generation_failure();
      },
      "Return the attached arena's generation-wide failure code.", py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
