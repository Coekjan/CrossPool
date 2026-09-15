#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/stl.h>

#include "bindings.hpp"
#include <xpool/kv/channel.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_kv(py::module_ &module) {
  auto kv = module.def_submodule("kv", "Host-local elastic KV-capacity coordination.");
  const auto construct_command = [](std::uint32_t sequence, std::uint32_t target_bundles,
                                    std::uint32_t active_bundles) {
    TORCH_CHECK(sequence != 0 && target_bundles != 0 && active_bundles != 0 && active_bundles <= target_bundles,
                "xpool kv command values are invalid");
    return xpool::kv::KvCapacityCommand{
        .sequence = sequence,
        .target_bundles = target_bundles,
        .active_bundles = active_bundles,
    };
  };

  py::class_<xpool::kv::KvCapacityCommand>(kv, "KvCapacityCommand", "One coherent group capacity command.")
      .def(py::init(construct_command), py::arg("sequence"), py::arg("target_bundles"), py::arg("active_bundles"))
      .def_readonly("sequence", &xpool::kv::KvCapacityCommand::sequence, "Monotonic command identity.")
      .def_readonly("target_bundles", &xpool::kv::KvCapacityCommand::target_bundles,
                    "Requested physical backing in bundles.")
      .def_readonly("active_bundles", &xpool::kv::KvCapacityCommand::active_bundles,
                    "Logically admitted backing in bundles.")
      .def(py::pickle(
          [](const xpool::kv::KvCapacityCommand &command) {
            return py::make_tuple(command.sequence, command.target_bundles, command.active_bundles);
          },
          [construct_command](const py::tuple &state) {
            TORCH_CHECK(state.size() == 3, "xpool kv command pickle must contain three fields");
            return construct_command(state[0].cast<std::uint32_t>(), state[1].cast<std::uint32_t>(),
                                     state[2].cast<std::uint32_t>());
          }));

  py::class_<xpool::kv::KvCapacityBackingReport>(kv, "KvCapacityBackingReport",
                                                 "One partition's physical backing progress.")
      .def(py::init([](std::uint32_t prepared_sequence, std::uint32_t backed_bundles) {
             TORCH_CHECK(backed_bundles != 0, "xpool kv backing report must retain a positive bundle prefix");
             return xpool::kv::KvCapacityBackingReport{
                 .prepared_sequence = prepared_sequence,
                 .backed_bundles = backed_bundles,
             };
           }),
           py::arg("prepared_sequence"), py::arg("backed_bundles"))
      .def_readonly("prepared_sequence", &xpool::kv::KvCapacityBackingReport::prepared_sequence,
                    "Latest command sequence physically prepared by the partition.")
      .def_readonly("backed_bundles", &xpool::kv::KvCapacityBackingReport::backed_bundles,
                    "Current mapped physical prefix in bundles.");

  py::class_<xpool::kv::KvCapacityPressureReport>(kv, "KvCapacityPressureReport",
                                                  "Latest group capacity-pressure publication.")
      .def_readonly("sequence", &xpool::kv::KvCapacityPressureReport::sequence,
                    "Monotonic pressure publication identity.")
      .def_readonly("active_bundles", &xpool::kv::KvCapacityPressureReport::active_bundles,
                    "Active capacity at rejection, or no value after resolution.");

  py::class_<xpool::kv::KvDeviceMemoryReport>(kv, "KvDeviceMemoryReport",
                                              "One attention GPU's post-capture memory observation.")
      .def_readonly("total_bytes", &xpool::kv::KvDeviceMemoryReport::total_bytes,
                    "Total physical device memory in bytes.")
      .def_readonly("free_bytes", &xpool::kv::KvDeviceMemoryReport::free_bytes,
                    "Free physical device memory after capture in bytes.");

  py::class_<xpool::kv::DaemonCapacityChannel>(kv, "DaemonCapacityChannel", "Daemon-owned elastic KV capacity channel.")
      .def_static(
          "create",
          [](std::size_t pool_count, std::size_t group_count, std::size_t partition_count) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Daemon,
                                                          "xpool.native.kv.DaemonCapacityChannel.create");
            return xpool::kv::DaemonCapacityChannel::create(pool_count, group_count, partition_count);
          },
          py::arg("pool_count"), py::arg("group_count"), py::arg("partition_count"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("name", &xpool::kv::DaemonCapacityChannel::name)
      .def("read_device_memory", &xpool::kv::DaemonCapacityChannel::read_device_memory,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_command", &xpool::kv::DaemonCapacityChannel::publish_command, py::arg("group_index"),
           py::arg("command"), py::call_guard<py::gil_scoped_release>())
      .def("read_backing_reports", &xpool::kv::DaemonCapacityChannel::read_backing_reports,
           py::call_guard<py::gil_scoped_release>())
      .def("read_pressure_reports", &xpool::kv::DaemonCapacityChannel::read_pressure_reports,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::kv::DaemonCapacityChannel::close, py::call_guard<py::gil_scoped_release>());

  py::class_<xpool::kv::AtnAgentCapacityChannel>(kv, "AtnAgentCapacityChannel",
                                                 "AtnAgent view of an elastic KV capacity channel.")
      .def_static(
          "attach",
          [](const std::string &name, std::size_t pool_index, std::vector<std::size_t> partition_indices) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                          "xpool.native.kv.AtnAgentCapacityChannel.attach");
            return xpool::kv::AtnAgentCapacityChannel::attach(name, pool_index, std::move(partition_indices));
          },
          py::arg("name"), py::arg("pool_index"), py::arg("partition_indices"),
          py::call_guard<py::gil_scoped_release>())
      .def("captures_complete", &xpool::kv::AtnAgentCapacityChannel::captures_complete,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_device_memory", &xpool::kv::AtnAgentCapacityChannel::publish_device_memory, py::arg("total_bytes"),
           py::arg("free_bytes"), py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::kv::AtnAgentCapacityChannel::close, py::call_guard<py::gil_scoped_release>());

  py::class_<xpool::kv::InstanceCapacityChannel>(kv, "InstanceCapacityChannel",
                                                 "Instance partition view of an elastic KV capacity channel.")
      .def_static(
          "attach",
          [](const std::string &name, std::size_t group_index, std::size_t partition_index, std::size_t group_count) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                          "xpool.native.kv.InstanceCapacityChannel.attach");
            return xpool::kv::InstanceCapacityChannel::attach(name, group_index, partition_index, group_count);
          },
          py::arg("name"), py::arg("group_index"), py::arg("partition_index"), py::arg("group_count"),
          py::call_guard<py::gil_scoped_release>())
      .def("publish_capture_complete", &xpool::kv::InstanceCapacityChannel::publish_capture_complete,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_backing_report", &xpool::kv::InstanceCapacityChannel::publish_backing_report, py::arg("report"),
           py::call_guard<py::gil_scoped_release>())
      .def("read_commands", &xpool::kv::InstanceCapacityChannel::read_commands,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_pressure", &xpool::kv::InstanceCapacityChannel::publish_pressure, py::arg("active_bundles"),
           py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::kv::InstanceCapacityChannel::close, py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
