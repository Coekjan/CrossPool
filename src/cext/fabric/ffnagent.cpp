#include <xpool/fabric/ffnagent.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <xpool/utils/layout.hpp>

namespace xpool::fabric {

namespace {

struct CoordinatorControlRegions {
  xpool::utils::layout::LayoutRegion activation_count;
  xpool::utils::layout::LayoutRegion coordinator_scheduler;
  xpool::utils::layout::LayoutRegion coordinator_scheduler_entries;
  std::size_t total_bytes;

  CoordinatorControlRegions(std::size_t instance_count) {
    using xpool::utils::layout::LayoutRegionSpec;
    const auto specs = std::to_array<LayoutRegionSpec>({
        LayoutRegionSpec::object<std::uint32_t>("FfnAgent activation count"),
        LayoutRegionSpec::object<Scheduler>("Fabric Coordinator Scheduler"),
        LayoutRegionSpec::array<SchedulerEntry>("Fabric Coordinator Scheduler entries", instance_count),
    });
    const auto plan = xpool::utils::layout::LayoutPlan{specs, alignof(std::uint64_t)};
    auto index = std::size_t{0};
    activation_count = plan[index++];
    coordinator_scheduler = plan[index++];
    coordinator_scheduler_entries = plan[index++];
    TORCH_CHECK(index == plan.regions.size(), "xpool FfnAgent control region plan is incomplete");
    total_bytes = plan.total_bytes;
  }
};

} // namespace

FfnAgentControl::~FfnAgentControl() {
  if (allocation_ != nullptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(allocation_));
  }
}

FfnAgentControl::FfnAgentControl(FfnAgentControl &&other) noexcept
    : allocation_(std::exchange(other.allocation_, nullptr)), view_(std::exchange(other.view_, {})) {}

FfnAgentControl &FfnAgentControl::operator=(FfnAgentControl &&other) {
  if (this != &other) {
    TORCH_CHECK(allocation_ == nullptr, "a live FfnAgent control allocation cannot be replaced by move");
    allocation_ = std::exchange(other.allocation_, nullptr);
    view_ = std::exchange(other.view_, {});
  }
  return *this;
}

std::size_t ffnagent_control_allocation_bytes(bool is_coordinator, std::size_t instance_count) {
  TORCH_CHECK(instance_count != 0, "xpool FfnAgent control requires a positive Instance count");
  return is_coordinator ? CoordinatorControlRegions{instance_count}.total_bytes : sizeof(std::uint32_t);
}

FfnAgentControl FfnAgentControl::create(bool is_coordinator, const SchedulerPolicy &scheduler_policy,
                                        std::size_t instance_count, std::size_t executor_lane_count) {
  TORCH_CHECK(instance_count != 0 && executor_lane_count != 0,
              "xpool FfnAgent control requires positive Instance and Executor Lane counts");
  auto control = FfnAgentControl{};
  if (!is_coordinator) {
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&control.allocation_), sizeof(std::uint32_t)));
    control.view_.activation_count = reinterpret_cast<std::uint32_t *>(control.allocation_);
    C10_CUDA_CHECK(cudaMemset(control.allocation_, 0, sizeof(std::uint32_t)));
    return control;
  }

  const auto regions = CoordinatorControlRegions{instance_count};
  C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&control.allocation_), regions.total_bytes));
  C10_CUDA_CHECK(cudaMemset(control.allocation_, 0, regions.total_bytes));
  control.view_.activation_count =
      reinterpret_cast<std::uint32_t *>(control.allocation_ + regions.activation_count.offset);
  control.view_.coordinator_scheduler =
      reinterpret_cast<Scheduler *>(control.allocation_ + regions.coordinator_scheduler.offset);
  auto *entries =
      reinterpret_cast<SchedulerEntry *>(control.allocation_ + regions.coordinator_scheduler_entries.offset);
  const auto coordinator_scheduler = Scheduler::from(scheduler_policy, entries, instance_count, executor_lane_count);
  C10_CUDA_CHECK(cudaMemcpy(control.view_.coordinator_scheduler, &coordinator_scheduler, sizeof(coordinator_scheduler),
                            cudaMemcpyHostToDevice));
  return control;
}

void FfnAgentControl::destroy() {
  if (allocation_ == nullptr) {
    return;
  }
  C10_CUDA_CHECK(cudaFree(allocation_));
  allocation_ = nullptr;
  view_ = {};
}

} // namespace xpool::fabric
