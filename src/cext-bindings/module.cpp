#include "bindings.hpp"

#include <c10/util/TypeCast.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <string>

#include <xpool/abi.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

PYBIND11_MODULE(native, module) {
  module.doc() = "Typed native control and trace bindings for xpool.";

  module.def(
      "abi_version", []() { return xpool::abi::kAbiVersion; },
      "Return the native ABI version compiled into this extension.");

  module.def(
      "initialize",
      [](std::int64_t role, const std::optional<std::int64_t> &cuda_device,
         const std::optional<std::string> &debug_options) {
        const auto parsed_role = xpool::RuntimeRole::parse(role);
        auto parsed_device = std::optional<c10::DeviceIndex>{};
        if (cuda_device.has_value()) {
          TORCH_CHECK(*cuda_device >= 0, "xpool initialize requires a non-negative CUDA device");
          parsed_device = c10::checked_convert<c10::DeviceIndex>(*cuda_device, "cuda_device");
        }
        xpool::RuntimeState::singleton().initialize(parsed_role, parsed_device);
        if (debug_options.has_value()) {
          TORCH_CHECK(parsed_role != xpool::RuntimeRole::Daemon,
                      "xpool daemon initialize requires null debug options");
          xpool::debug::configure(xpool::debug::DebugOptions::parse(*debug_options), *parsed_device);
        }
      },
      py::arg("role"), py::arg("cuda_device") = py::none(), py::arg("debug_options") = py::none(),
      "Initialize the process role, optional CUDA device, and debug options.");

  xpool::bindings::bind_fabric(module);
  xpool::bindings::bind_transport(module);
}
