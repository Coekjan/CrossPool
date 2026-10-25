#include <xpool/bindings.hpp>

#include <torch/python.h>

#include <xpool/ffn.hpp>

namespace xpool::bindings {

c10::ScalarType require_payload_dtype(const pybind11::object &dtype, std::string_view context) {
  const auto value = torch::python::detail::py_object_to_dtype(dtype);
  TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(value), "xpool ", context,
              " payload dtype must be torch.float16 or torch.bfloat16");
  return value;
}

} // namespace xpool::bindings
