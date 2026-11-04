#include <cstddef>

#include <pybind11/pybind11.h>

#include <xpool/bindings.hpp>
#include <xpool/debug/options.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_debug(py::module_ &module) {
  auto debug = module.def_submodule("debug", "Native debug options.");
  py::class_<xpool::debug::TraceObserverOptions>(debug, "TraceObserverOptions", "Immutable native trace options.")
      .def(py::init([](bool enable, std::size_t record_capacity) {
             return xpool::debug::TraceObserverOptions{enable, record_capacity};
           }),
           py::arg("enable"), py::arg("record_capacity"));

  py::class_<xpool::debug::GraphObserverOptions>(debug, "GraphObserverOptions",
                                                 "Immutable native Graph Observer options.")
      .def(py::init([](bool enable) { return xpool::debug::GraphObserverOptions{enable}; }), py::arg("enable"));

  py::class_<xpool::debug::FfnRoutingObserverOptions>(debug, "FfnRoutingObserverOptions",
                                                      "Immutable native FFN Routing Observer options.")
      .def(py::init([](bool enable, std::size_t record_capacity) {
             return xpool::debug::FfnRoutingObserverOptions{enable, record_capacity};
           }),
           py::arg("enable"), py::arg("record_capacity"));

  py::class_<xpool::debug::Options>(debug, "Options", "Immutable native debug-options snapshot.")
      .def(py::init([](xpool::debug::TraceObserverOptions transport_observer,
                       xpool::debug::TraceObserverOptions fabric_observer,
                       xpool::debug::GraphObserverOptions graph_observer,
                       xpool::debug::FfnRoutingObserverOptions ffn_routing_observer) {
             auto options = xpool::debug::Options{};
             options.transport_observer = transport_observer;
             options.fabric_observer = fabric_observer;
             options.graph_observer = graph_observer;
             options.ffn_routing_observer = ffn_routing_observer;
             return options;
           }),
           py::arg("transport_observer"), py::arg("fabric_observer"), py::arg("graph_observer"),
           py::arg("ffn_routing_observer"));
}

} // namespace xpool::bindings
