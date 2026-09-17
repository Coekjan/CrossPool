#include <cstddef>
#include <cstdint>
#include <limits>
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
  auto kv = module.def_submodule("kv", "Host-local elastic KV control.");
  const auto construct_command = [](std::uint32_t sequence, std::uint32_t target_bundles) {
    TORCH_CHECK(sequence != 0 && target_bundles != 0, "xpool kv command values are invalid");
    return xpool::kv::KvCapacityCommand{.sequence = sequence, .target_bundles = target_bundles};
  };

  py::class_<xpool::kv::KvCapacityCommand>(kv, "KvCapacityCommand", "One immutable group capacity operation.")
      .def(py::init(construct_command), py::arg("sequence"), py::arg("target_bundles"))
      .def_readonly("sequence", &xpool::kv::KvCapacityCommand::sequence, "Monotonic operation identity.")
      .def_readonly("target_bundles", &xpool::kv::KvCapacityCommand::target_bundles,
                    "Absolute physical and logical bundle target.")
      .def(py::pickle(
          [](const xpool::kv::KvCapacityCommand &command) {
            return py::make_tuple(command.sequence, command.target_bundles);
          },
          [construct_command](const py::tuple &state) {
            TORCH_CHECK(state.size() == 2, "xpool kv command pickle must contain two fields");
            return construct_command(state[0].cast<std::uint32_t>(), state[1].cast<std::uint32_t>());
          }));

  py::class_<xpool::kv::KvCapacityDemand>(kv, "KvCapacityDemand", "Latest capacity-correlated admission demand.")
      .def(py::init([](std::uint32_t evaluated_sequence, std::optional<std::uint32_t> requested_bundles,
                       std::optional<std::uint64_t> deadline_monotonic_ns) {
             TORCH_CHECK(
                 evaluated_sequence != 0 && requested_bundles.has_value() == deadline_monotonic_ns.has_value() &&
                     (!requested_bundles || *requested_bundles != 0) &&
                     (!deadline_monotonic_ns || (*deadline_monotonic_ns != 0 &&
                                                 *deadline_monotonic_ns != std::numeric_limits<std::uint64_t>::max())),
                 "xpool kv demand values are invalid");
             return xpool::kv::KvCapacityDemand{
                 .evaluated_sequence = evaluated_sequence,
                 .requested_bundles = requested_bundles,
                 .deadline_monotonic_ns = deadline_monotonic_ns,
             };
           }),
           py::arg("evaluated_sequence"), py::arg("requested_bundles"), py::arg("deadline_monotonic_ns"))
      .def_readonly("evaluated_sequence", &xpool::kv::KvCapacityDemand::evaluated_sequence,
                    "Completed operation whose capacity was evaluated.")
      .def_readonly("requested_bundles", &xpool::kv::KvCapacityDemand::requested_bundles,
                    "Absolute requested capacity, or none after resolution.")
      .def_readonly("deadline_monotonic_ns", &xpool::kv::KvCapacityDemand::deadline_monotonic_ns,
                    "Scheduler-local absolute SLO deadline, or none after resolution.");

  py::class_<xpool::kv::KvCapacityCompletion>(kv, "KvCapacityCompletion", "One partition's terminal operation result.")
      .def(py::init([](std::uint32_t sequence, std::uint32_t backed_bundles) {
             TORCH_CHECK(sequence != 0 && backed_bundles != 0, "xpool kv completion values are invalid");
             return xpool::kv::KvCapacityCompletion{.sequence = sequence, .backed_bundles = backed_bundles};
           }),
           py::arg("sequence"), py::arg("backed_bundles"))
      .def_readonly("sequence", &xpool::kv::KvCapacityCompletion::sequence, "Completed operation identity.")
      .def_readonly("backed_bundles", &xpool::kv::KvCapacityCompletion::backed_bundles,
                    "Actual physical backing after applying the command.");

  py::class_<xpool::kv::KvDeviceMemoryReport>(kv, "KvDeviceMemoryReport",
                                              "One attention GPU's post-capture memory observation.")
      .def_readonly("total_bytes", &xpool::kv::KvDeviceMemoryReport::total_bytes,
                    "Total physical device memory in bytes.")
      .def_readonly("free_bytes", &xpool::kv::KvDeviceMemoryReport::free_bytes,
                    "Free physical device memory after capture in bytes.");

  py::class_<xpool::kv::DaemonControlChannel>(kv, "DaemonControlChannel", "Daemon-owned elastic KV control channel.")
      .def_static(
          "create",
          [](std::size_t pool_count, std::size_t group_count, std::size_t partition_count) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Daemon,
                                                          "xpool.native.kv.DaemonControlChannel.create");
            return xpool::kv::DaemonControlChannel::create(pool_count, group_count, partition_count);
          },
          py::arg("pool_count"), py::arg("group_count"), py::arg("partition_count"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("name", &xpool::kv::DaemonControlChannel::name)
      .def("read_device_memory", &xpool::kv::DaemonControlChannel::read_device_memory,
           py::call_guard<py::gil_scoped_release>())
      .def("read_initial_backing", &xpool::kv::DaemonControlChannel::read_initial_backing,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_service_ceiling", &xpool::kv::DaemonControlChannel::publish_service_ceiling, py::arg("group_index"),
           py::arg("bundles"), py::call_guard<py::gil_scoped_release>())
      .def("publish_command", &xpool::kv::DaemonControlChannel::publish_command, py::arg("group_index"),
           py::arg("command"), py::call_guard<py::gil_scoped_release>())
      .def("read_demands", &xpool::kv::DaemonControlChannel::read_demands, py::call_guard<py::gil_scoped_release>())
      .def("read_completions", &xpool::kv::DaemonControlChannel::read_completions,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::kv::DaemonControlChannel::close, py::call_guard<py::gil_scoped_release>());

  py::class_<xpool::kv::AtnAgentControlChannel>(kv, "AtnAgentControlChannel",
                                                "AtnAgent view of an elastic KV control channel.")
      .def_static(
          "attach",
          [](const std::string &name, std::size_t pool_index, std::vector<std::size_t> partition_indices) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                          "xpool.native.kv.AtnAgentControlChannel.attach");
            return xpool::kv::AtnAgentControlChannel::attach(name, pool_index, std::move(partition_indices));
          },
          py::arg("name"), py::arg("pool_index"), py::arg("partition_indices"),
          py::call_guard<py::gil_scoped_release>())
      .def("captures_complete", &xpool::kv::AtnAgentControlChannel::captures_complete,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_device_memory", &xpool::kv::AtnAgentControlChannel::publish_device_memory, py::arg("total_bytes"),
           py::arg("free_bytes"), py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::kv::AtnAgentControlChannel::close, py::call_guard<py::gil_scoped_release>());

  py::class_<xpool::kv::InstanceControlChannel>(kv, "InstanceControlChannel",
                                                "Instance partition view of an elastic KV control channel.")
      .def_static(
          "attach",
          [](const std::string &name, std::size_t group_index, std::size_t partition_index,
             std::size_t dp_group_count) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                          "xpool.native.kv.InstanceControlChannel.attach");
            return xpool::kv::InstanceControlChannel::attach(name, group_index, partition_index, dp_group_count);
          },
          py::arg("name"), py::arg("group_index"), py::arg("partition_index"), py::arg("dp_group_count"),
          py::call_guard<py::gil_scoped_release>())
      .def("publish_initial_backing", &xpool::kv::InstanceControlChannel::publish_initial_backing, py::arg("bundles"),
           py::call_guard<py::gil_scoped_release>())
      .def("publish_capture_complete", &xpool::kv::InstanceControlChannel::publish_capture_complete,
           py::call_guard<py::gil_scoped_release>())
      .def("service_ceiling", &xpool::kv::InstanceControlChannel::service_ceiling,
           py::call_guard<py::gil_scoped_release>())
      .def("read_commands", &xpool::kv::InstanceControlChannel::read_commands, py::call_guard<py::gil_scoped_release>())
      .def("publish_completion", &xpool::kv::InstanceControlChannel::publish_completion, py::arg("completion"),
           py::call_guard<py::gil_scoped_release>())
      .def("publish_demand", &xpool::kv::InstanceControlChannel::publish_demand, py::arg("demand"),
           py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::kv::InstanceControlChannel::close, py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
