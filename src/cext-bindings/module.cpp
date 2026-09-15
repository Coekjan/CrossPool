#include <cstdint>
#include <optional>

#include <c10/util/TypeCast.h>
#include <pybind11/native_enum.h>
#include <pybind11/stl.h>

#include "bindings.hpp"
#include <xpool/abi.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/ffn.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

PYBIND11_MODULE(native, module) {
  module.doc() = "Typed native control and trace bindings for CrossPool.";

  module.attr("ABI_VERSION") = xpool::abi::kVersion;

  py::native_enum<xpool::RuntimeRole>(module, "RuntimeRole", "enum.IntEnum", "Native process role.")
      .value("DAEMON", xpool::RuntimeRole::Daemon)
      .value("INSTANCE", xpool::RuntimeRole::Instance)
      .value("ATNAGENT", xpool::RuntimeRole::AtnAgent)
      .value("FFNAGENT", xpool::RuntimeRole::FfnAgent)
      .finalize();

  auto ffn = module.def_submodule("ffn", "Shared native FFN semantics.");
  py::native_enum<xpool::ffn::ForwardMode>(ffn, "ForwardMode", "enum.IntEnum")
      .value("PREFILL", xpool::ffn::ForwardMode::Prefill)
      .value("DECODE", xpool::ffn::ForwardMode::Decode)
      .value("IDLE", xpool::ffn::ForwardMode::Idle)
      .finalize();
  py::native_enum<xpool::ffn::OutputRequirement>(ffn, "OutputRequirement", "enum.IntEnum")
      .value("PER_RANK_COMPLETE", xpool::ffn::OutputRequirement::PerRankComplete)
      .value("GROUP_SUM_COMPLETE", xpool::ffn::OutputRequirement::GroupSumComplete)
      .finalize();
  py::native_enum<xpool::ffn::DpRowLayout>(ffn, "DpRowLayout", "enum.IntEnum")
      .value("NONE", xpool::ffn::DpRowLayout::None)
      .value("UNIFORM_BY_RANK", xpool::ffn::DpRowLayout::UniformByRank)
      .value("PACKED_BY_RANK", xpool::ffn::DpRowLayout::PackedByRank)
      .finalize();
  py::native_enum<xpool::ffn::ResultCode>(ffn, "ResultCode", "enum.IntEnum")
      .value("OK", xpool::ffn::ResultCode::Ok)
      .value("SHUTDOWN", xpool::ffn::ResultCode::Shutdown)
      .value("PROTOCOL_MISMATCH", xpool::ffn::ResultCode::ProtocolMismatch)
      .value("TIMEOUT", xpool::ffn::ResultCode::Timeout)
      .finalize();
  py::native_enum<xpool::ffn::LayerKind>(ffn, "LayerKind", "enum.IntEnum")
      .value("DENSE", xpool::ffn::LayerKind::Dense)
      .value("MOE", xpool::ffn::LayerKind::Moe)
      .finalize();

  xpool::bindings::bind_debug(module);

  module.def(
      "initialize",
      [](xpool::RuntimeRole role, const std::optional<std::int64_t> &cuda_device,
         const std::optional<xpool::debug::Options> &debug_options) {
        auto parsed_device = std::optional<c10::DeviceIndex>{};
        if (cuda_device.has_value()) {
          TORCH_CHECK(*cuda_device >= 0, "xpool initialize requires a non-negative CUDA device");
          parsed_device = c10::checked_convert<c10::DeviceIndex>(*cuda_device, "cuda_device");
        }
        TORCH_CHECK(xpool::is_valid(role), "xpool initialize received an invalid runtime role");
        xpool::RuntimeState::singleton().initialize(role, parsed_device);
        xpool::debug::configure(debug_options.value_or(xpool::debug::Options{}), parsed_device);
      },
      py::arg("role"), py::arg("cuda_device") = py::none(), py::arg("debug_options") = py::none(),
      "Initialize the process role, optional CUDA device, and debug options.");

  module.def(
      "runtime_role", []() { return xpool::RuntimeState::singleton().role(); }, "Return the initialized process role.");

  xpool::bindings::bind_fabric(module);
  xpool::bindings::bind_transport(module);
  xpool::bindings::bind_devkit(module);
}
