#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include <cstddef>

#include <xpool/debug/options.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_debug(py::module_ &module) {
  auto debug = module.def_submodule("debug", "Native debug options.");
  py::class_<xpool::debug::TraceObserverOptions>(debug, "TraceObserverOptions", "Immutable native trace options.")
      .def(py::init([](bool enable, std::size_t record_capacity) {
             return xpool::debug::TraceObserverOptions{enable, record_capacity};
           }),
           py::arg("enable"), py::arg("record_capacity"))
      .def_readonly("enable", &xpool::debug::TraceObserverOptions::enable, "Whether trace collection is enabled.")
      .def_readonly("record_capacity", &xpool::debug::TraceObserverOptions::record_capacity,
                    "Configured native trace capacity.");

  py::class_<xpool::debug::GraphObserverOptions>(debug, "GraphObserverOptions",
                                                 "Immutable native Graph Observer options.")
      .def(py::init([](bool enable) { return xpool::debug::GraphObserverOptions{enable}; }), py::arg("enable"))
      .def_readonly("enable", &xpool::debug::GraphObserverOptions::enable,
                    "Whether the observed Graph snapshot may be read.");

  py::class_<xpool::debug::FfnRoutingObserverOptions>(debug, "FfnRoutingObserverOptions",
                                                      "Immutable native FFN Routing Observer options.")
      .def(py::init([](bool enable, std::size_t record_capacity) {
             return xpool::debug::FfnRoutingObserverOptions{enable, record_capacity};
           }),
           py::arg("enable"), py::arg("record_capacity"))
      .def_readonly("enable", &xpool::debug::FfnRoutingObserverOptions::enable, "Whether routing records are retained.")
      .def_readonly("record_capacity", &xpool::debug::FfnRoutingObserverOptions::record_capacity,
                    "Maximum retained routing records.");

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
           py::arg("ffn_routing_observer"))
      .def_readonly("transport_observer", &xpool::debug::Options::transport_observer, "Transport trace options.")
      .def_readonly("fabric_observer", &xpool::debug::Options::fabric_observer, "Fabric trace options.")
      .def_readonly("graph_observer", &xpool::debug::Options::graph_observer, "Graph Observer options.")
      .def_readonly("ffn_routing_observer", &xpool::debug::Options::ffn_routing_observer,
                    "FFN Routing Observer options.");
}

} // namespace xpool::bindings
