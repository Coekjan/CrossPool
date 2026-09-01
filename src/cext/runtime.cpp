#include <xpool/runtime.hpp>

#include <c10/util/Exception.h>

#include <algorithm>

namespace xpool {

std::string_view name(RuntimeRole role) {
  switch (role) {
  case RuntimeRole::Daemon:
    return "daemon";
  case RuntimeRole::Instance:
    return "instance";
  case RuntimeRole::AtnAgent:
    return "atnagent";
  case RuntimeRole::FfnAgent:
    return "ffnagent";
  }
  TORCH_CHECK(false, "xpool received an invalid runtime role");
}

void RuntimeState::initialize(RuntimeRole role, const std::optional<c10::DeviceIndex> &cuda_device) {
  if (role == RuntimeRole::Daemon) {
    TORCH_CHECK(!cuda_device.has_value(), "xpool daemon init requires a null CUDA device");
  } else {
    TORCH_CHECK(cuda_device.has_value() && *cuda_device >= 0,
                "xpool GPU runtime init requires a non-negative CUDA device");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (role_.has_value()) {
    TORCH_CHECK(*role_ == role, "xpool op init requires runtime role ", name(role),
                " but current process was initialized as ", name(*role_));
    TORCH_CHECK(device_ == cuda_device, "xpool init CUDA device differs from the first init call");
  }
  role_ = role;
  device_ = cuda_device;
}

void RuntimeState::require_role(std::initializer_list<RuntimeRole> expected, std::string_view op_name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(role_.has_value(), "xpool op ", op_name, " requires xpool.init first");
  const auto matches = std::find(expected.begin(), expected.end(), *role_) != expected.end();
  if (!matches && expected.size() == 1) {
    TORCH_CHECK(false, "xpool op ", op_name, " requires runtime role ", name(*expected.begin()),
                " but current process was initialized as ", name(*role_));
  }
  TORCH_CHECK(matches, "xpool op ", op_name,
              " requires one of the accepted runtime roles but current "
              "process was initialized as ",
              name(*role_));
}

c10::DeviceIndex RuntimeState::cuda_device(std::string_view op_name) const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(device_.has_value(), "xpool op ", op_name, " requires xpool.init first");
  return *device_;
}

} // namespace xpool
