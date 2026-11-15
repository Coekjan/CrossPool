#include <xpool/utils/cuda.hpp>

#include <cstdint>

#include <xpool/bindings.hpp>
#include <xpool/utils/cuda/graph.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_cuda(py::module_ &module) {
  auto cuda = module.def_submodule("cuda", "CUDA stream owners and borrowed resource observations.");
  py::class_<xpool::utils::cuda::OwnedCudaStream>(
      cuda, "OwnedCudaStream", "Exclusive CUDA stream owner retained through borrowed graph retirement.")
      .def_static(
          "create", []() { return xpool::utils::cuda::OwnedCudaStream::create(); },
          "Create an exclusive nonblocking stream on the current device.")
      .def_property_readonly(
          "address",
          [](const xpool::utils::cuda::OwnedCudaStream &owner) {
            return reinterpret_cast<std::uintptr_t>(owner.get());
          },
          "Borrow the stream address while the owner remains live.")
      .def("synchronize", &xpool::utils::cuda::OwnedCudaStream::synchronize,
           "Confirm completion before releasing capture resources.")
      .def("destroy", &xpool::utils::cuda::OwnedCudaStream::destroy,
           "Destroy after graph retirement; preserve the handle on failure.");

  auto graph = cuda.def_submodule("graph", "CUDA graph interval observation.");
  graph.def(
      "references_range",
      [](std::uintptr_t graph_address, std::uintptr_t address, std::size_t bytes) {
        return xpool::utils::cuda::graph::references_range(reinterpret_cast<cudaGraph_t>(graph_address), address,
                                                           bytes);
      },
      py::arg("graph_address"), py::arg("address"), py::arg("bytes"),
      "Locate declared interval references and validate typed accesses.");
}

} // namespace xpool::bindings
