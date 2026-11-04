#include <xpool/utils/cuda.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <xpool/bindings.hpp>
#include <xpool/utils/cuda/blas.hpp>
#include <xpool/utils/cuda/graph.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_cuda(py::module_ &module) {
  auto cuda = module.def_submodule("cuda", "CUDA stream owners and borrowed resource observations.");
  py::class_<xpool::utils::cuda::ResourceCapacity>(cuda, "ResourceCapacity",
                                                   "Address-free resource capacity, alignment and backing aliases.")
      .def_readonly("key", &xpool::utils::cuda::ResourceCapacity::key,
                    "Stable source role independent of allocation addresses.")
      .def_readonly("max_bytes", &xpool::utils::cuda::ResourceCapacity::max_bytes,
                    "Positive prospective logical capacity in bytes.")
      .def_readonly("alignment_bytes", &xpool::utils::cuda::ResourceCapacity::alignment_bytes,
                    "Required positive power-of-two byte alignment.")
      .def_readonly("aliases", &xpool::utils::cuda::ResourceCapacity::aliases,
                    "Unique source identities sharing this canonical backing.");

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
           "Destroy after graph retirement and source-cache cleanup; preserve the handle on failure.");

  py::class_<xpool::utils::cuda::CaptureAllocation>(cuda, "CaptureAllocation",
                                                    "Borrowed source interval valid through synchronous installation.")
      .def(py::init<std::string, std::uintptr_t, std::size_t>(), py::arg("key"), py::arg("address"), py::arg("bytes"))
      .def_readonly("key", &xpool::utils::cuda::CaptureAllocation::key,
                    "Source role normalized through registered backing aliases.")
      .def_readonly("address", &xpool::utils::cuda::CaptureAllocation::address,
                    "Borrowed device base; the capture scope retains its owner.")
      .def_readonly("bytes", &xpool::utils::cuda::CaptureAllocation::bytes,
                    "Positive logical byte extent of the source backing.");

  auto blas = cuda.def_submodule("blas", "ATen BLAS workspace discovery and lifecycle.");
  blas.def(
      "probe_capacity",
      [](const xpool::utils::cuda::OwnedCudaStream &stream) {
        return xpool::utils::cuda::blas::probe_capacity(stream.get());
      },
      py::arg("stream"), "Discover capacity on a fresh owned stream and release probe owners.");
  blas.def(
      "snapshot_workspaces",
      [](const xpool::utils::cuda::OwnedCudaStream &stream) {
        return xpool::utils::cuda::blas::snapshot_workspaces(stream.get());
      },
      py::arg("stream"), "Borrow observed workspace ranges while their owners remain live.");
  blas.def(
      "release_workspaces",
      [](const xpool::utils::cuda::OwnedCudaStream &stream) {
        xpool::utils::cuda::blas::release_workspaces(stream.get());
      },
      py::arg("stream"), "Synchronize and release stream-local workspace cache owners.");
  auto graph = cuda.def_submodule("graph", "CUDA graph interval observation.");
  graph.def(
      "used_resource_regions",
      [](std::uintptr_t address, const std::vector<xpool::utils::cuda::CaptureAllocation> &allocations) {
        return xpool::utils::cuda::graph::used_resource_regions(reinterpret_cast<cudaGraph_t>(address), allocations);
      },
      py::arg("graph_address"), py::arg("allocations"),
      "Locate declared interval references and validate typed accesses.");
}

} // namespace xpool::bindings
