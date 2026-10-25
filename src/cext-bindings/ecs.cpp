#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/stl.h>
#include <pybind11/typing.h>

#include <xpool/bindings.hpp>
#include <xpool/ecs/channel.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_ecs(py::module_ &module) {
  auto ecs = module.def_submodule("ecs", "Host-local Elastic Context State byte-quota control.");
  const auto construct_command = [](std::uint32_t sequence,
                                    py::typing::Tuple<std::uint64_t, py::ellipsis> quota_bytes) {
    TORCH_CHECK(sequence != 0 && !quota_bytes.empty(), "xpool elastic context state command values are invalid");
    return xpool::ecs::QuotaCommand{.sequence = sequence,
                                    .quota_bytes = quota_bytes.cast<std::vector<std::uint64_t>>()};
  };
  py::class_<xpool::ecs::QuotaCommand>(ecs, "QuotaCommand", "One immutable per-TP-rank byte grant.")
      .def(py::init(construct_command), py::arg("sequence"), py::arg("quota_bytes"))
      .def_readonly("sequence", &xpool::ecs::QuotaCommand::sequence, "Monotonic operation identity.")
      .def_property_readonly(
          "quota_bytes", [](const xpool::ecs::QuotaCommand &command) { return typed_tuple(command.quota_bytes); },
          "Absolute physical-byte permission in TP-rank order.")
      .def(py::pickle(
          [](const xpool::ecs::QuotaCommand &command) {
            return py::make_tuple(command.sequence, typed_tuple(command.quota_bytes));
          },
          [construct_command](const py::tuple &state) {
            TORCH_CHECK(state.size() == 2, "xpool elastic context state command pickle must contain two fields");
            return construct_command(state[0].cast<std::uint32_t>(),
                                     state[1].cast<py::typing::Tuple<std::uint64_t, py::ellipsis>>());
          }));

  py::class_<xpool::ecs::QuotaDemand>(ecs, "QuotaDemand", "Latest grant-correlated admission demand.")
      .def(py::init([](std::uint32_t evaluated_sequence,
                       std::optional<py::typing::Tuple<std::uint64_t, py::ellipsis>> required_bytes,
                       std::optional<std::uint64_t> deadline_monotonic_ns) {
             TORCH_CHECK(evaluated_sequence != 0 && required_bytes.has_value() == deadline_monotonic_ns.has_value() &&
                             (!required_bytes || !required_bytes->empty()),
                         "xpool elastic context state demand values are invalid");
             return xpool::ecs::QuotaDemand{
                 .evaluated_sequence = evaluated_sequence,
                 .required_bytes =
                     required_bytes ? std::optional{required_bytes->cast<std::vector<std::uint64_t>>()} : std::nullopt,
                 .deadline_monotonic_ns = deadline_monotonic_ns,
             };
           }),
           py::arg("evaluated_sequence"), py::arg("required_bytes"), py::arg("deadline_monotonic_ns"))
      .def_readonly("evaluated_sequence", &xpool::ecs::QuotaDemand::evaluated_sequence,
                    "Completed grant whose capacity was evaluated.")
      .def_property_readonly(
          "required_bytes",
          [](const xpool::ecs::QuotaDemand &demand) -> std::optional<py::typing::Tuple<std::uint64_t, py::ellipsis>> {
            return demand.required_bytes ? std::optional{typed_tuple(*demand.required_bytes)} : std::nullopt;
          },
          "Absolute per-rank backing requirements, or none after resolution.")
      .def_readonly("deadline_monotonic_ns", &xpool::ecs::QuotaDemand::deadline_monotonic_ns,
                    "Original absolute scheduler-local deadline, or none after resolution.");

  py::class_<xpool::ecs::QuotaCompletion>(ecs, "QuotaCompletion", "One rank's terminal quota adoption.")
      .def(py::init([](std::uint32_t sequence, std::uint64_t backed_bytes) {
             TORCH_CHECK(sequence != 0, "xpool elastic context state completion sequence is invalid");
             return xpool::ecs::QuotaCompletion{.sequence = sequence, .backed_bytes = backed_bytes};
           }),
           py::arg("sequence"), py::arg("backed_bytes"))
      .def_readonly("sequence", &xpool::ecs::QuotaCompletion::sequence, "Completed operation identity.")
      .def_readonly("backed_bytes", &xpool::ecs::QuotaCompletion::backed_bytes,
                    "Actual retained physical bytes after quota adoption and retirement.");

  py::class_<xpool::ecs::DeviceMemoryReport>(ecs, "DeviceMemoryReport",
                                             "One attention device's post-capture memory observation.")
      .def_readonly("total_bytes", &xpool::ecs::DeviceMemoryReport::total_bytes,
                    "Total physical device memory in bytes.")
      .def_readonly("free_bytes", &xpool::ecs::DeviceMemoryReport::free_bytes,
                    "Free physical device memory after capture in bytes.");

  py::class_<xpool::ecs::DaemonControlChannel>(ecs, "DaemonControlChannel",
                                               "Daemon-owned Elastic Context State quota channel.")
      .def_static(
          "create",
          [](std::size_t device_count, py::typing::Tuple<std::size_t, py::ellipsis> group_partition_counts) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Daemon,
                                                          "xpool.native.ecs.DaemonControlChannel.create");
            return xpool::ecs::DaemonControlChannel::create(device_count,
                                                            group_partition_counts.cast<std::vector<std::size_t>>());
          },
          py::arg("device_count"), py::arg("group_partition_counts"))
      .def_property_readonly("name", &xpool::ecs::DaemonControlChannel::name)
      .def("read_device_memory", &xpool::ecs::DaemonControlChannel::read_device_memory,
           py::call_guard<py::gil_scoped_release>())
      .def("read_initial_backing", &xpool::ecs::DaemonControlChannel::read_initial_backing,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "publish_service_ceiling",
          [](xpool::ecs::DaemonControlChannel &channel, std::size_t group_index,
             py::typing::Tuple<std::uint64_t, py::ellipsis> quota_bytes) {
            channel.publish_service_ceiling(group_index, quota_bytes.cast<std::vector<std::uint64_t>>());
          },
          py::arg("group_index"), py::arg("quota_bytes"))
      .def("publish_command", &xpool::ecs::DaemonControlChannel::publish_command, py::arg("group_index"),
           py::arg("command"), py::call_guard<py::gil_scoped_release>())
      .def("read_demands", &xpool::ecs::DaemonControlChannel::read_demands, py::call_guard<py::gil_scoped_release>())
      .def("read_completions", &xpool::ecs::DaemonControlChannel::read_completions,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::ecs::DaemonControlChannel::close, py::call_guard<py::gil_scoped_release>());

  py::class_<xpool::ecs::AtnAgentControlChannel>(ecs, "AtnAgentControlChannel",
                                                 "AtnAgent view of an Elastic Context State quota channel.")
      .def_static(
          "attach",
          [](const std::string &name, std::size_t device_index, std::vector<std::size_t> partition_indices) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                          "xpool.native.ecs.AtnAgentControlChannel.attach");
            return xpool::ecs::AtnAgentControlChannel::attach(name, device_index, std::move(partition_indices));
          },
          py::arg("name"), py::arg("device_index"), py::arg("partition_indices"),
          py::call_guard<py::gil_scoped_release>())
      .def("captures_complete", &xpool::ecs::AtnAgentControlChannel::captures_complete,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_device_memory", &xpool::ecs::AtnAgentControlChannel::publish_device_memory, py::arg("total_bytes"),
           py::arg("free_bytes"), py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::ecs::AtnAgentControlChannel::close, py::call_guard<py::gil_scoped_release>());

  py::class_<xpool::ecs::InstanceControlChannel>(ecs, "InstanceControlChannel",
                                                 "Instance-rank view of an Elastic Context State quota channel.")
      .def_static(
          "attach",
          [](const std::string &name, std::size_t group_index, std::size_t partition_index,
             std::size_t dp_group_count) {
            xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                          "xpool.native.ecs.InstanceControlChannel.attach");
            return xpool::ecs::InstanceControlChannel::attach(name, group_index, partition_index, dp_group_count);
          },
          py::arg("name"), py::arg("group_index"), py::arg("partition_index"), py::arg("dp_group_count"),
          py::call_guard<py::gil_scoped_release>())
      .def("publish_initial_backing", &xpool::ecs::InstanceControlChannel::publish_initial_backing,
           py::arg("backed_bytes"), py::call_guard<py::gil_scoped_release>())
      .def("service_ceiling",
           [](const xpool::ecs::InstanceControlChannel &channel)
               -> std::optional<py::typing::Tuple<std::uint64_t, py::ellipsis>> {
             const auto ceiling = channel.service_ceiling();
             return ceiling ? std::optional{typed_tuple(*ceiling)} : std::nullopt;
           })
      .def("read_commands", &xpool::ecs::InstanceControlChannel::read_commands,
           py::call_guard<py::gil_scoped_release>())
      .def("publish_completion", &xpool::ecs::InstanceControlChannel::publish_completion, py::arg("completion"),
           py::call_guard<py::gil_scoped_release>())
      .def("publish_demand", &xpool::ecs::InstanceControlChannel::publish_demand, py::arg("demand"),
           py::call_guard<py::gil_scoped_release>())
      .def("close", &xpool::ecs::InstanceControlChannel::close, py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
