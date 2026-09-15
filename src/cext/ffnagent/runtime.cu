#include <xpool/ffnagent/runtime.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda_runtime.h>
#include <nvshmem.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/fabric/protocol.cuh>
#include <xpool/fabric/runtime.hpp>
#include <xpool/ffn.hpp>
#include <xpool/ffnagent/delivery.cuh>
#include <xpool/ffnagent/hooks.hpp>
#include <xpool/ffnagent/parameterization.hpp>
#include <xpool/ffnagent/runtime.cuh>
#include <xpool/hooks.cuh>
#include <xpool/hooks.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/graph.hpp>
#include <xpool/utils/wait.cuh>

namespace xpool::ffnagent {

namespace {

constexpr auto kLaneControlBlockSize = 256U;
constexpr auto kDeliveryBlocksPerMultiprocessor = 4U;
constexpr auto kDirectDeliveryBranch = std::uint32_t{0};
constexpr auto kCompleteDeliveryBranch = std::uint32_t{1};
constexpr auto kDeliveryBranchCount = std::uint32_t{2};

void validate_cuda_address(std::uintptr_t address, int expected_device, const char *name) {
  TORCH_CHECK(address != 0, "xpool ", name, " has a null CUDA address");
  auto attributes = cudaPointerAttributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, reinterpret_cast<const void *>(address)));
  TORCH_CHECK(attributes.type == cudaMemoryTypeDevice && attributes.device == expected_device, "xpool ", name,
              " is not Device memory on CUDA device ", expected_device);
}

LayerBindingValues materialize_binding_values(const BindingResourceProjection &resources) {
  return std::visit(
      [](const auto &value) {
        using Resource = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<Resource, DenseBindingResourceProjection>) {
          return LayerBindingValues{
              .gate_up_weight_address = value.gate_up_weight_address,
              .down_weight_address = value.down_weight_address,
              .router_weight_address = 0,
              .router_correction_bias_address = 0,
          };
        } else {
          return LayerBindingValues{
              .gate_up_weight_address = value.expert_gate_up_weight_address,
              .down_weight_address = value.expert_down_weight_address,
              .router_weight_address = value.router.has_value() ? value.router->weight_address : 0,
              .router_correction_bias_address =
                  value.router.has_value() && value.router->correction_bias_address.has_value()
                      ? *value.router->correction_bias_address
                      : 0,
          };
        }
      },
      resources);
}

std::vector<ResourceReplacement> make_resource_replacements(const ExecutionSignatureProjection &signature) {
  return std::visit(
      [&](const auto &value) {
        using Signature = std::remove_cvref_t<decltype(value)>;
        auto result = std::vector<ResourceReplacement>{};
        if constexpr (std::is_same_v<Signature, DenseExecutionSignatureProjection>) {
          result = {
              ResourceReplacement{value.primary_capture_resources.gate_up_weight_address,
                                  value.control_capture_resources.gate_up_weight_address,
                                  value.primary_capture_resources.gate_up_weight_address,
                                  offsetof(LayerBindingValues, gate_up_weight_address)},
              ResourceReplacement{value.primary_capture_resources.down_weight_address,
                                  value.control_capture_resources.down_weight_address,
                                  value.primary_capture_resources.down_weight_address,
                                  offsetof(LayerBindingValues, down_weight_address)},
          };
        } else {
          result = {
              ResourceReplacement{value.primary_capture_resources.expert_gate_up_weight_address,
                                  value.control_capture_resources.expert_gate_up_weight_address,
                                  value.primary_capture_resources.expert_gate_up_weight_address,
                                  offsetof(LayerBindingValues, gate_up_weight_address)},
              ResourceReplacement{value.primary_capture_resources.expert_down_weight_address,
                                  value.control_capture_resources.expert_down_weight_address,
                                  value.primary_capture_resources.expert_down_weight_address,
                                  offsetof(LayerBindingValues, down_weight_address)},
          };
          if (value.primary_capture_resources.router.has_value()) {
            result.push_back(ResourceReplacement{
                value.primary_capture_resources.router->weight_address,
                value.control_capture_resources.router->weight_address,
                value.primary_capture_resources.router->weight_address,
                offsetof(LayerBindingValues, router_weight_address),
            });
            if (value.primary_capture_resources.router->correction_bias_address.has_value()) {
              result.push_back(ResourceReplacement{
                  *value.primary_capture_resources.router->correction_bias_address,
                  *value.control_capture_resources.router->correction_bias_address,
                  *value.primary_capture_resources.router->correction_bias_address,
                  offsetof(LayerBindingValues, router_correction_bias_address),
              });
            }
          }
        }
        return result;
      },
      signature);
}

} // namespace

namespace {

void validate_layer_resource_targets(const ExecutionProjection &projection, int device) {
  const auto validate_resources = [device](const auto &resources) {
    using Resource = std::remove_cvref_t<decltype(resources)>;
    if constexpr (std::is_same_v<Resource, DenseBindingResourceProjection>) {
      validate_cuda_address(resources.gate_up_weight_address, device, "Dense gate/up weight");
      validate_cuda_address(resources.down_weight_address, device, "Dense down weight");
    } else {
      validate_cuda_address(resources.expert_gate_up_weight_address, device, "MoE Expert gate/up weights");
      validate_cuda_address(resources.expert_down_weight_address, device, "MoE Expert down weights");
      if (resources.router.has_value()) {
        validate_cuda_address(resources.router->weight_address, device, "MoE Router weight");
        if (resources.router->correction_bias_address.has_value()) {
          validate_cuda_address(*resources.router->correction_bias_address, device, "MoE Router correction bias");
        }
      }
    }
  };

  for (const auto &layer : projection.layers) {
    std::visit(validate_resources, layer.layer_resource_targets);
  }
}

} // namespace

} // namespace xpool::ffnagent

namespace xpool::ffnagent {

namespace {

XPOOL_DEVICE_FN void terminate_lane(xpool::fabric::ArenaView arena, LaneRuntimeState &state,
                                    cudaGraphConditionalHandle while_handle, cudaGraphConditionalHandle compute_handle,
                                    std::uint32_t compute_empty_branch_index, xpool::ffn::ResultCode result) {
  state.result_code = result;
  state.terminal = true;
  cudaGraphSetConditional(compute_handle, compute_empty_branch_index);
  cudaGraphSetConditional(while_handle, 0);
  if (result == xpool::ffn::ResultCode::ProtocolMismatch || result == xpool::ffn::ResultCode::Timeout) {
    arena.state().failure.try_publish(arena.layout().coordinator_pe(), result, state.execution.key,
                                      state.execution.layer_ordinal);
  }
}

XPOOL_DEVICE_FN void terminate_delivery(xpool::fabric::ArenaView arena, LaneRuntimeState &state,
                                        cudaGraphConditionalHandle while_handle,
                                        cudaGraphConditionalHandle observation_handle, xpool::ffn::ResultCode result) {
  state.result_code = result;
  state.terminal = true;
  cudaGraphSetConditional(observation_handle, 0);
  cudaGraphSetConditional(while_handle, 0);
  if (result == xpool::ffn::ResultCode::ProtocolMismatch || result == xpool::ffn::ResultCode::Timeout) {
    arena.state().failure.try_publish(arena.layout().coordinator_pe(), result, state.execution.key,
                                      state.execution.layer_ordinal);
  }
}

XPOOL_KERNEL_FN void publish_lane_activation(std::uint32_t *activation_count) {
  cuda::atomic_ref{*activation_count}.fetch_add(std::uint32_t{1}, cuda::memory_order_release);
}

XPOOL_KERNEL_FN void wait_pre_compute(xpool::fabric::ArenaView arena, std::size_t executor_lane_index,
                                      cudaGraphConditionalHandle while_handle,
                                      cudaGraphConditionalHandle compute_handle, LaneRuntimeState *state,
                                      const LayerExecutionEntry *layer_entries, std::size_t layer_entry_count,
                                      const CapacityExecutionEntry *capacity_entries, std::size_t signature_count,
                                      std::uint32_t compute_empty_branch_index) {
  cudaGraphSetConditional(compute_handle, compute_empty_branch_index);
  state->terminal = false;
  state->result_code = xpool::ffn::ResultCode::Ok;

  // Phase: Observe Execution - A Lane accepts only the next monotonic lease;
  // shutdown or canonical failure terminates its resident loop.
  auto &execution_ready = arena.lane_execution_publication(executor_lane_index);
  const auto execution_result = xpool::utils::wait::until(
      xpool::utils::wait::Deadline::never(),
      [&] { return execution_ready.test_at_least(state->last_lease_sequence + 1); },
      [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
  if (execution_result != xpool::utils::wait::Status::Ready) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   arena.cancellation_result());
    return;
  }

  state->execution = execution_ready.record;
  if (execution_ready.validate_expected(state->execution.executor_lease_sequence, state->execution.key) !=
          xpool::ffn::ResultCode::Ok ||
      state->execution.key.instance_index >= arena.layout().instance_count) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   xpool::ffn::ResultCode::ProtocolMismatch);
    return;
  }

  // Phase: Select Compute Branch - Resolve the requested model layer and the
  // smallest installed Capacity that contains its live rows.
  auto layer_index = layer_entry_count;
  for (auto index = std::size_t{0}; index < layer_entry_count; ++index) {
    if (layer_entries[index].instance_index == state->execution.key.instance_index &&
        layer_entries[index].layer_ordinal == state->execution.layer_ordinal) {
      layer_index = index;
      break;
    }
  }
  if (layer_index == layer_entry_count) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   xpool::ffn::ResultCode::ProtocolMismatch);
    return;
  }
  const auto &layer = layer_entries[layer_index];
  auto selected = static_cast<const CapacityExecutionEntry *>(nullptr);
  for (auto offset = std::size_t{0}; offset < layer.capacity_count; ++offset) {
    const auto &candidate = capacity_entries[layer.capacity_begin + offset];
    if (state->execution.payload_rows <= candidate.payload_row_capacity) {
      selected = &candidate;
      break;
    }
  }
  if (selected == nullptr || selected->execution_signature_index >= signature_count) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   xpool::ffn::ResultCode::ProtocolMismatch);
    return;
  }

  // Phase: Resolve TP Membership - Derive this PE's immutable rank in the
  // layer's planned FfnAgent group before exposing the selected execution.
  const auto &instance = arena.instance_entry(state->execution.key.instance_index);
  const auto ffnagent_pes = arena.ffnagent_pes(state->execution.key.instance_index, state->execution.layer_ordinal);
  const auto pe = nvshmem_my_pe();
  const auto member = cuda::std::find(ffnagent_pes.begin(), ffnagent_pes.end(), pe);
  if (member == ffnagent_pes.end() || ffnagent_pes.size() != instance.ffn_tp_size) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   xpool::ffn::ResultCode::ProtocolMismatch);
    return;
  }
  const auto local_tp_rank = static_cast<std::size_t>(member - ffnagent_pes.begin());

  state->selected_payload_row_capacity = selected->payload_row_capacity;
  state->selected_execution_signature_index = selected->execution_signature_index;
  state->selected_layer_entry_index = layer_index;
  state->local_tp_rank = local_tp_rank;
  xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
      {.arena = arena,
       .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::LaneExecutionObserved,
       .execution = &state->execution,
       .executor_lane_index = executor_lane_index,
       .payload_row_capacity = selected->payload_row_capacity,
       .delivery = xpool::fabric::delivery_variant(instance, state->execution.output_requirement)});

  // Phase: Await Payloads - Compute becomes selectable only after input and,
  // for MoE, routing publications agree with the active Lane lease.
  const auto deadline = xpool::utils::wait::Deadline::after(xpool::utils::wait::kDefaultTimeoutNanoseconds);
  auto &input_ready = arena.input_ready_publication(executor_lane_index);
  auto protocol_valid = true;
  const auto result = xpool::utils::wait::until(
      deadline,
      [&] {
        const auto input_observed = input_ready.test_at_least(state->execution.executor_lease_sequence);
        if (input_observed && input_ready.validate_expected(state->execution.executor_lease_sequence,
                                                            state->execution.key) != xpool::ffn::ResultCode::Ok) {
          protocol_valid = false;
          return true;
        }
        auto routing_observed = true;
        if (layer.requires_routing_metadata) {
          auto &routing_ready = arena.routing_metadata_ready_publication(executor_lane_index);
          routing_observed = routing_ready.test_at_least(state->execution.executor_lease_sequence);
          if (routing_observed && routing_ready.validate_expected(state->execution.executor_lease_sequence,
                                                                  state->execution.key) != xpool::ffn::ResultCode::Ok) {
            protocol_valid = false;
            return true;
          }
        }
        return input_observed && routing_observed;
      },
      [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
  if (!protocol_valid) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   xpool::ffn::ResultCode::ProtocolMismatch);
    return;
  }
  if (result != xpool::utils::wait::Status::Ready) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   result == xpool::utils::wait::Status::TimedOut ? xpool::ffn::ResultCode::Timeout
                                                                  : arena.cancellation_result());
    return;
  }
  xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
      {.arena = arena,
       .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::InputReadyObserved,
       .execution = &state->execution,
       .executor_lane_index = executor_lane_index});
  if (layer.requires_routing_metadata) {
    xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
        {.arena = arena,
         .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataObserved,
         .execution = &state->execution,
         .executor_lane_index = executor_lane_index});
  }
}

XPOOL_KERNEL_FN void bind_compute(xpool::fabric::ArenaView arena, std::size_t executor_lane_index,
                                  cudaGraphConditionalHandle while_handle, cudaGraphConditionalHandle compute_handle,
                                  LaneRuntimeState *state, const LayerExecutionEntry *layer_entries,
                                  const DiscoveredBindingSchema *schemas, std::uint32_t compute_empty_branch_index,
                                  const BindingSite *sites, cudaGraphKernelNodeUpdate *updates) {
  if (state->terminal) {
    return;
  }
  if (arena.shutdown_requested() || arena.state().failure.published()) {
    terminate_lane(arena, *state, while_handle, compute_handle, compute_empty_branch_index,
                   arena.cancellation_result());
    return;
  }
  const auto &layer = layer_entries[state->selected_layer_entry_index];
  const auto &schema = schemas[state->selected_execution_signature_index];
  for (auto index = std::size_t{0}; index < schema.site_count; ++index) {
    const auto &site = sites[schema.site_begin + index];
    updates[index].node = site.node;
    updates[index].field = cudaGraphKernelNodeFieldParam;
    updates[index].updateData.param.pValue =
        reinterpret_cast<const unsigned char *>(&layer.binding_values) + site.value_offset_bytes;
    updates[index].updateData.param.offset = site.parameter_offset_bytes;
    updates[index].updateData.param.size = sizeof(std::uintptr_t);
  }
  xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
      {.arena = arena,
       .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeStarted,
       .execution = &state->execution,
       .executor_lane_index = executor_lane_index});
  const auto update_result =
      schema.site_count == 0 ? cudaSuccess : cudaGraphKernelNodeUpdatesApply(updates, schema.site_count);
  xpool::abort_if(update_result != cudaSuccess);
  cudaGraphSetConditional(compute_handle, static_cast<std::uint32_t>(state->selected_execution_signature_index));
}

XPOOL_KERNEL_FN void publish_routing_metadata(xpool::fabric::ArenaView arena, std::size_t executor_lane_index,
                                              std::size_t payload_row_capacity, LaneRuntimeState *state) {
  const auto group = cooperative_groups::this_thread_block();
  XPOOL_DEVICE_SHARED bool terminal;
  if (group.thread_rank() == 0) {
    terminal = state->terminal || arena.shutdown_requested() || arena.state().failure.published();
    state->terminal = terminal;
  }
  group.sync();
  if (terminal) {
    return;
  }
  const auto ffnagent_pes = arena.ffnagent_pes(state->execution.key.instance_index, state->execution.layer_ordinal);
  xpool::abort_if(ffnagent_pes.empty() || ffnagent_pes[0] != nvshmem_my_pe() || state->local_tp_rank != 0);
  auto &publication = arena.routing_metadata_ready_publication(executor_lane_index);
  if (group.thread_rank() == 0) {
    publication.record = xpool::fabric::RoutingMetadataReady{
        .key = state->execution.key,
        .executor_lease_sequence = state->execution.executor_lease_sequence,
    };
  }
  group.sync();
  auto metadata = arena.routing_metadata(executor_lane_index, payload_row_capacity);
  for (auto rank = std::size_t{1}; rank < ffnagent_pes.size(); ++rank) {
    publication.publish_payload(group, ffnagent_pes[rank], metadata.publication_payload());
  }
  xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
      {.arena = arena,
       .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataPublished,
       .execution = &state->execution,
       .executor_lane_index = executor_lane_index,
       .payload_row_capacity = payload_row_capacity});
}

XPOOL_KERNEL_FN void prepare_delivery(xpool::fabric::ArenaView arena, std::size_t executor_lane_index,
                                      cudaGraphConditionalHandle while_handle,
                                      cudaGraphConditionalHandle delivery_handle,
                                      cudaGraphConditionalHandle peer_partial_handle, LaneRuntimeState *state) {
  const auto group = cooperative_groups::this_thread_block();
  extern XPOOL_DEVICE_SHARED int reader_pes[];
  if (group.thread_rank() == 0) {
    cudaGraphSetConditional(delivery_handle, kDeliveryBranchCount);
    cudaGraphSetConditional(peer_partial_handle, 0);
    state->peer_partial_wait_started_at = 0;
  }
  group.sync();
  if (state->terminal || arena.shutdown_requested() || arena.state().failure.published()) {
    if (group.thread_rank() == 0) {
      state->terminal = true;
      cudaGraphSetConditional(while_handle, 0);
    }
    return;
  }
  if (group.thread_rank() == 0) {
    xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
        {.arena = arena,
         .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeCompleted,
         .execution = &state->execution,
         .executor_lane_index = executor_lane_index});
  }
  group.sync();

  const auto &execution = state->execution;
  const auto &instance = arena.instance_entry(execution.key.instance_index);
  const auto ffnagent_pes = arena.ffnagent_pes(execution.key.instance_index, execution.layer_ordinal);
  const auto delivery = xpool::fabric::delivery_variant(instance, execution.output_requirement);
  if (delivery == xpool::fabric::DeliveryVariant::DirectPartial) {
    if (group.thread_rank() == 0) {
      cudaGraphSetConditional(delivery_handle, kDirectDeliveryBranch);
    }
    return;
  }

  const auto active_reader_count =
      execution.payload_rows < ffnagent_pes.size() ? execution.payload_rows : ffnagent_pes.size();
  XPOOL_DEVICE_SHARED std::size_t destination_count;
  if (group.thread_rank() == 0) {
    destination_count = 0;
    for (auto rank = std::size_t{0}; rank < active_reader_count; ++rank) {
      if (rank != state->local_tp_rank) {
        reader_pes[destination_count++] = ffnagent_pes[rank];
      }
    }
    auto &partial_ready = arena.partial_ready_publication(
        static_cast<std::size_t>(nvshmem_my_pe()) - arena.layout().atnagent_count, executor_lane_index);
    partial_ready.record = xpool::fabric::PartialReady{
        .key = execution.key,
        .executor_lease_sequence = execution.executor_lease_sequence,
    };
  }
  group.sync();
  auto &partial_ready = arena.partial_ready_publication(
      static_cast<std::size_t>(nvshmem_my_pe()) - arena.layout().atnagent_count, executor_lane_index);
  partial_ready.publish_after_local_payload(group, cuda::std::span<const int>{reader_pes, destination_count});
  if (group.thread_rank() == 0 && destination_count != 0) {
    xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
        {.arena = arena,
         .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PartialReadyPublished,
         .execution = &state->execution,
         .executor_lane_index = executor_lane_index});
  }
  group.sync();
  if (group.thread_rank() == 0) {
    const auto active_reader = state->local_tp_rank < active_reader_count;
    if (active_reader && ffnagent_pes.size() > 1) {
      state->peer_partial_wait_started_at = xpool::utils::time::now();
      cudaGraphSetConditional(peer_partial_handle, 1);
    }
    cudaGraphSetConditional(delivery_handle, active_reader ? kCompleteDeliveryBranch : kDeliveryBranchCount);
  }
}

XPOOL_KERNEL_FN void probe_peer_partials(xpool::fabric::ArenaView arena, std::size_t executor_lane_index,
                                         cudaGraphConditionalHandle while_handle,
                                         cudaGraphConditionalHandle observation_handle, LaneRuntimeState *state) {
  const auto ffnagent_pes = arena.ffnagent_pes(state->execution.key.instance_index, state->execution.layer_ordinal);
  auto ready = true;
  for (auto rank = std::size_t{0}; rank < ffnagent_pes.size(); ++rank) {
    if (rank == state->local_tp_rank) {
      continue;
    }
    const auto source_index = static_cast<std::size_t>(ffnagent_pes[rank]) - arena.layout().atnagent_count;
    auto &peer = arena.partial_ready_publication(source_index, executor_lane_index);
    const auto observed = peer.test_at_least(state->execution.executor_lease_sequence);
    if (observed && peer.validate_expected(state->execution.executor_lease_sequence, state->execution.key) !=
                        xpool::ffn::ResultCode::Ok) {
      terminate_delivery(arena, *state, while_handle, observation_handle, xpool::ffn::ResultCode::ProtocolMismatch);
      return;
    }
    ready = ready && observed;
  }
  const auto deadline = xpool::utils::wait::Deadline::from_start(state->peer_partial_wait_started_at,
                                                                 xpool::utils::wait::kDefaultTimeoutNanoseconds);
  const auto result = xpool::utils::wait::poll_once(
      deadline, [&] { return ready; }, [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
  if (result == xpool::utils::wait::Status::Pending) {
    xpool::utils::wait::relax();
    cudaGraphSetConditional(observation_handle, 1);
    return;
  }
  if (result != xpool::utils::wait::Status::Ready) {
    terminate_delivery(arena, *state, while_handle, observation_handle,
                       result == xpool::utils::wait::Status::TimedOut ? xpool::ffn::ResultCode::Timeout
                                                                      : arena.cancellation_result());
    return;
  }
  xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
      {.arena = arena,
       .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PeerPartialsReadyObserved,
       .execution = &state->execution,
       .executor_lane_index = executor_lane_index});
  cudaGraphSetConditional(observation_handle, 0);
}

XPOOL_KERNEL_FN void complete_lane_execution(xpool::fabric::ArenaView arena, std::size_t executor_lane_index,
                                             cudaGraphConditionalHandle while_handle, LaneRuntimeState *state) {
  const auto group = cooperative_groups::this_thread_block();
  if (state->terminal || arena.shutdown_requested() || arena.state().failure.published()) {
    group.sync();
    if (group.thread_rank() == 0) {
      nvshmem_quiet();
      state->terminal = true;
      cudaGraphSetConditional(while_handle, 0);
    }
    return;
  }

  const auto local_ffnagent_index = static_cast<std::size_t>(nvshmem_my_pe()) - arena.layout().atnagent_count;
  auto &completion = arena.ffnagent_completion_publication(local_ffnagent_index, executor_lane_index);
  if (group.thread_rank() == 0) {
    completion.record = xpool::fabric::FfnAgentCompletion{
        .key = state->execution.key,
        .executor_lease_sequence = state->execution.executor_lease_sequence,
    };
  }
  group.sync();
  completion.publish_after_remote_payload(group, arena.layout().coordinator_pe());
  if (group.thread_rank() == 0) {
    xpool::hooks::FabricFfnAgentProtocolEvent::hooks(
        {.arena = arena,
         .kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind::CompletionPublished,
         .execution = &state->execution,
         .executor_lane_index = executor_lane_index});
    state->last_lease_sequence = state->execution.executor_lease_sequence;
    state->peer_partial_wait_started_at = 0;
    cudaGraphSetConditional(while_handle, 1);
  }
}

void insert_routing_publication(const PrimaryGraphLocation &location, xpool::fabric::ArenaView arena,
                                std::size_t executor_lane_index, std::size_t payload_row_capacity,
                                LaneRuntimeState *state) {
  auto dependent_count = std::size_t{0};
  C10_CUDA_CHECK(cudaGraphNodeGetDependentNodes(location.node, nullptr, nullptr, &dependent_count));
  TORCH_CHECK(dependent_count != 0, "xpool Routing Finalization has no Expert successor");
  void *arguments[] = {&arena, &executor_lane_index, &payload_row_capacity, &state};
  const auto publication =
      xpool::utils::graph::add_kernel_node(location.graph, reinterpret_cast<const void *>(publish_routing_metadata),
                                           dim3{1, 1, 1}, dim3{kLaneControlBlockSize, 1, 1}, 0, arguments);
  // Routing metadata becomes remotely visible after finalization and before
  // every Expert consumer that previously depended directly on that node.
  xpool::utils::graph::insert_node_after(location.graph, location.node, publication);
}

} // namespace

ExecutionRuntime::LaneOwner::~LaneOwner() {
  try {
    destroy();
  } catch (...) {
  }
}

ExecutionRuntime::LaneOwner::LaneOwner(LaneOwner &&other) noexcept
    : graph(std::exchange(other.graph, nullptr)), executable(std::exchange(other.executable, nullptr)),
      stream(std::move(other.stream)), state(std::exchange(other.state, nullptr)),
      schemas(std::exchange(other.schemas, nullptr)), sites(std::exchange(other.sites, nullptr)),
      updates(std::exchange(other.updates, nullptr)), workspace(std::exchange(other.workspace, nullptr)),
      workspace_bytes(std::exchange(other.workspace_bytes, 0)) {}

ExecutionRuntime::LaneOwner &ExecutionRuntime::LaneOwner::operator=(LaneOwner &&other) noexcept {
  if (this != &other) {
    xpool::abort_if(graph != nullptr || executable != nullptr || stream || state != nullptr || schemas != nullptr ||
                    sites != nullptr || updates != nullptr || workspace != nullptr);
    graph = std::exchange(other.graph, nullptr);
    executable = std::exchange(other.executable, nullptr);
    stream = std::move(other.stream);
    state = std::exchange(other.state, nullptr);
    schemas = std::exchange(other.schemas, nullptr);
    sites = std::exchange(other.sites, nullptr);
    updates = std::exchange(other.updates, nullptr);
    workspace = std::exchange(other.workspace, nullptr);
    workspace_bytes = std::exchange(other.workspace_bytes, 0);
  }
  return *this;
}

void ExecutionRuntime::LaneOwner::destroy() {
  if (executable != nullptr) {
    C10_CUDA_CHECK(cudaGraphExecDestroy(executable));
    executable = nullptr;
  }
  if (graph != nullptr) {
    C10_CUDA_CHECK(cudaGraphDestroy(graph));
    graph = nullptr;
  }
  for (auto **pointer : {&state, &schemas, &sites, &updates}) {
    if (*pointer != nullptr) {
      C10_CUDA_CHECK(cudaFree(*pointer));
      *pointer = nullptr;
    }
  }
  if (workspace != nullptr) {
    C10_CUDA_CHECK(cudaFree(workspace));
    workspace = nullptr;
    workspace_bytes = 0;
  }
  stream.destroy();
}

ExecutionRuntime::~ExecutionRuntime() {
  xpool::abort_if(installed_ || active_ || layer_entries_ != nullptr || capacity_entries_ != nullptr ||
                  !lanes_.empty());
}

void ExecutionRuntime::materialize_shared_tables(const ExecutionProjection &projection,
                                                 const xpool::fabric::ArenaProjection &fabric_projection,
                                                 std::size_t ffnagent_index) {
  auto host_layers = std::vector<LayerExecutionEntry>{};
  auto host_capacities = std::vector<CapacityExecutionEntry>{};
  auto local_layer_index = std::size_t{0};
  for (auto instance_index = std::size_t{0}; instance_index < fabric_projection.instances.size(); ++instance_index) {
    const auto &instance = fabric_projection.instances[instance_index];
    const auto maximum_capacity = std::max(instance.decode_payload_row_capacity, instance.prefill_payload_row_capacity);
    for (auto layer_ordinal = std::size_t{0}; layer_ordinal < instance.layers.size(); ++layer_ordinal) {
      const auto &fabric_layer = instance.layers[layer_ordinal];
      const auto member =
          std::find(fabric_layer.ffnagent_indices.begin(), fabric_layer.ffnagent_indices.end(), ffnagent_index);
      if (member == fabric_layer.ffnagent_indices.end()) {
        continue;
      }
      TORCH_CHECK(local_layer_index < projection.layers.size(),
                  "xpool execution Projection omits a local Fabric layer");
      const auto &layer = projection.layers[local_layer_index++];
      TORCH_CHECK(layer.instance_index == instance_index && layer.layer_ordinal == layer_ordinal,
                  "xpool execution Projection local layers are not co-indexed with Fabric");
      const auto local_tp_rank = static_cast<std::size_t>(std::distance(fabric_layer.ffnagent_indices.begin(), member));
      const auto router_owner = fabric_layer.kind == xpool::ffn::LayerKind::Moe && local_tp_rank == 0;
      const auto values = materialize_binding_values(layer.layer_resource_targets);
      const auto capacity_begin = host_capacities.size();
      for (const auto signature_index : layer.execution_signature_indices) {
        const auto &signature = projection.signatures[signature_index];
        std::visit(
            [&](const auto &value) {
              using Signature = std::remove_cvref_t<decltype(value)>;
              constexpr auto moe = std::is_same_v<Signature, MoeExecutionSignatureProjection>;
              TORCH_CHECK(value.payload_dtype == instance.payload_dtype && value.hidden_size == instance.hidden_size,
                          "xpool Execution Signature payload geometry disagrees with Fabric");
              TORCH_CHECK(moe == (fabric_layer.kind == xpool::ffn::LayerKind::Moe),
                          "xpool Execution Signature kind disagrees with Fabric");
              if constexpr (moe) {
                TORCH_CHECK(value.effective_topk == fabric_layer.effective_topk &&
                                value.routed_expert_count.has_value() == router_owner,
                            "xpool MoE Signature disagrees with Fabric topology");
              }
            },
            signature);
        host_capacities.push_back(CapacityExecutionEntry{
            .payload_row_capacity = std::visit([](const auto &value) { return value.payload_row_capacity; }, signature),
            .execution_signature_index = signature_index,
        });
      }
      TORCH_CHECK(host_capacities.back().payload_row_capacity == maximum_capacity,
                  "xpool local layer maximum Capacity disagrees with Fabric");
      host_layers.push_back(LayerExecutionEntry{
          .instance_index = instance_index,
          .layer_ordinal = layer_ordinal,
          .capacity_begin = capacity_begin,
          .capacity_count = layer.execution_signature_indices.size(),
          .binding_values = values,
          .requires_routing_metadata = fabric_layer.kind == xpool::ffn::LayerKind::Moe && !router_owner,
      });
    }
  }
  TORCH_CHECK(local_layer_index == projection.layers.size(),
              "xpool execution Projection contains a nonlocal Fabric layer");
  layer_entry_count_ = host_layers.size();
  if (!host_layers.empty()) {
    const auto bytes = xpool::utils::checked::prod(host_layers.size(), sizeof(LayerExecutionEntry));
    C10_CUDA_CHECK(cudaMalloc(&layer_entries_, bytes));
    C10_CUDA_CHECK(cudaMemcpy(layer_entries_, host_layers.data(), bytes, cudaMemcpyHostToDevice));
  }
  if (!host_capacities.empty()) {
    const auto bytes = xpool::utils::checked::prod(host_capacities.size(), sizeof(CapacityExecutionEntry));
    C10_CUDA_CHECK(cudaMalloc(&capacity_entries_, bytes));
    C10_CUDA_CHECK(cudaMemcpy(capacity_entries_, host_capacities.data(), bytes, cudaMemcpyHostToDevice));
  }
}

void ExecutionRuntime::materialize_lane(const ExecutionProjection &projection, xpool::fabric::ArenaView arena,
                                        const xpool::fabric::ArenaLayout &layout,
                                        const xpool::fabric::ArenaProjection &fabric_projection,
                                        std::size_t ffnagent_index, std::size_t executor_lane_index, int device,
                                        std::uint32_t *activation_count) {
  // Phase: Size Lane Resources - Derive the largest workspace, delivery grid,
  // and TP reader scratch needed by any execution selectable on this Lane.
  auto maximum_workspace_bytes = std::size_t{0};
  for (const auto &signature : projection.signatures) {
    maximum_workspace_bytes =
        std::max(maximum_workspace_bytes,
                 std::visit([](const auto &value) { return value.compute_workspace_bytes; }, signature));
  }
  auto multiprocessor_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  const auto delivery_block_count = xpool::utils::checked::prod(
      static_cast<std::size_t>(multiprocessor_count), static_cast<std::size_t>(kDeliveryBlocksPerMultiprocessor));
  TORCH_CHECK(delivery_block_count <= std::numeric_limits<unsigned int>::max(),
              "xpool FFN delivery grid exceeds the CUDA dimension domain");
  auto maximum_local_tp_size = std::size_t{0};
  for (const auto &instance : fabric_projection.instances) {
    for (const auto &layer : instance.layers) {
      if (std::find(layer.ffnagent_indices.begin(), layer.ffnagent_indices.end(), ffnagent_index) !=
          layer.ffnagent_indices.end()) {
        maximum_local_tp_size = std::max(maximum_local_tp_size, layer.ffnagent_indices.size());
      }
    }
  }
  const auto reader_scratch_bytes = xpool::utils::checked::prod(maximum_local_tp_size, sizeof(int));
  TORCH_CHECK(reader_scratch_bytes <= std::numeric_limits<unsigned int>::max(),
              "xpool FFN reader scratch exceeds CUDA shared memory");

  // Phase: Allocate Lane State - Every Lane owns an independently launchable
  // GraphExec, stream, workspace, and stable Device state.
  const auto signature_count = projection.signatures.size();
  const auto compute_switch_size = static_cast<std::uint32_t>(std::max(signature_count, std::size_t{1}));
  const auto compute_empty_branch_index = static_cast<std::uint32_t>(signature_count);
  auto *device_layer_entries = static_cast<LayerExecutionEntry *>(layer_entries_);
  auto *device_capacity_entries = static_cast<CapacityExecutionEntry *>(capacity_entries_);
  lanes_.emplace_back();
  auto &lane = lanes_.back();
  lane.stream = xpool::utils::device::OwnedCudaStream::create();
  if (maximum_workspace_bytes != 0) {
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lane.workspace), maximum_workspace_bytes));
    lane.workspace_bytes = maximum_workspace_bytes;
  }
  C10_CUDA_CHECK(cudaMalloc(&lane.state, sizeof(LaneRuntimeState)));
  const auto initial_state = LaneRuntimeState{};
  C10_CUDA_CHECK(cudaMemcpy(lane.state, &initial_state, sizeof(initial_state), cudaMemcpyHostToDevice));
  auto *device_state = static_cast<LaneRuntimeState *>(lane.state);

  // Phase: Embed Compute Branches - Build the resident outer WHILE and embed
  // every captured Signature behind one Device-selected switch.
  C10_CUDA_CHECK(cudaGraphCreate(&lane.graph, 0));
  auto while_handle = xpool::utils::graph::create_conditional_handle(lane.graph, 1, cudaGraphCondAssignDefault);
  const auto while_node = xpool::utils::graph::add_while_node(lane.graph, while_handle);
  const auto while_body = xpool::utils::graph::conditional_bodies(while_node).front();

  void *activation_arguments[] = {&activation_count};
  const auto activation_node =
      xpool::utils::graph::add_kernel_node(lane.graph, reinterpret_cast<const void *>(publish_lane_activation),
                                           dim3{1, 1, 1}, dim3{1, 1, 1}, 0, activation_arguments);
  xpool::utils::graph::add_dependency(lane.graph, activation_node, while_node);

  auto compute_handle = xpool::utils::graph::create_conditional_handle(while_body);
  const auto compute_node = xpool::utils::graph::add_switch_node(while_body, compute_handle, compute_switch_size);
  const auto compute_bodies = xpool::utils::graph::conditional_bodies(compute_node);

  auto host_schemas = std::vector<DiscoveredBindingSchema>{};
  auto host_sites = std::vector<BindingSite>{};
  host_schemas.reserve(signature_count);
  auto maximum_site_count = std::size_t{0};
  for (auto signature_index = std::size_t{0}; signature_index < signature_count; ++signature_index) {
    const auto &signature = projection.signatures[signature_index];
    const auto primary_graph = std::visit([](const auto &value) { return value.primary_graph_address; }, signature);
    const auto embedded_graph = xpool::utils::graph::embed_child_graph(compute_bodies[signature_index],
                                                                       reinterpret_cast<cudaGraph_t>(primary_graph));

    // Captured placeholder addresses become this Lane's stable Fabric and
    // workspace addresses; weight sites remain Device-updatable per layer.
    const auto input_target = reinterpret_cast<std::uintptr_t>(arena.ffn_input_storage(layout, executor_lane_index));
    const auto partial_target =
        reinterpret_cast<std::uintptr_t>(arena.ffn_partial_storage(layout, executor_lane_index));
    const auto workspace_target = reinterpret_cast<std::uintptr_t>(lane.workspace);
    auto capture_routing_weights_address = std::uintptr_t{0};
    auto routing_target = std::uintptr_t{0};
    auto lane_replacements = std::vector<LaneAddressReplacement>{};
    std::visit(
        [&](const auto &value) {
          const auto io_bytes =
              xpool::utils::checked::prod(value.payload_row_capacity, value.hidden_size,
                                          static_cast<std::size_t>(c10::elementSize(value.payload_dtype)));
          lane_replacements = {
              LaneAddressReplacement{
                  .capture_address = value.capture_input_address,
                  .bytes = io_bytes,
                  .target_address = input_target,
              },
              LaneAddressReplacement{
                  .capture_address = value.capture_partial_address,
                  .bytes = io_bytes,
                  .target_address = partial_target,
              },
              LaneAddressReplacement{
                  .capture_address = value.capture_workspace_address,
                  .bytes = value.compute_workspace_bytes,
                  .target_address = workspace_target,
              },
          };
          using Signature = std::remove_cvref_t<decltype(value)>;
          if constexpr (std::is_same_v<Signature, MoeExecutionSignatureProjection>) {
            const auto route_elements = xpool::utils::checked::prod(value.payload_row_capacity, value.effective_topk);
            const auto route_bytes =
                xpool::utils::checked::prod(route_elements, std::size_t{sizeof(std::int32_t) + sizeof(float)});
            TORCH_CHECK(layout.routing_metadata_offset_bytes != 0 &&
                            layout.routing_metadata_stride_bytes >= route_bytes,
                        "xpool Fabric lacks projected Routing Metadata storage");
            routing_target =
                reinterpret_cast<std::uintptr_t>(arena.routing_metadata_storage(layout, executor_lane_index));
            capture_routing_weights_address =
                value.capture_routing_metadata_address + route_elements * sizeof(std::int32_t);
            lane_replacements.push_back(LaneAddressReplacement{
                .capture_address = value.capture_routing_metadata_address,
                .bytes = route_bytes,
                .target_address = routing_target,
            });
            if (value.capture_payload_rows_address.has_value()) {
              lane_replacements.push_back(LaneAddressReplacement{
                  .capture_address = *value.capture_payload_rows_address,
                  .bytes = sizeof(std::int64_t),
                  .target_address =
                      reinterpret_cast<std::uintptr_t>(arena.lane_payload_rows(layout, executor_lane_index)),
              });
            }
          }
        },
        signature);
    const auto resources = make_resource_replacements(signature);
    const auto capture_routing_metadata_address = std::visit(
        [](const auto &value) -> std::uintptr_t {
          using Signature = std::remove_cvref_t<decltype(value)>;
          if constexpr (std::is_same_v<Signature, MoeExecutionSignatureProjection>) {
            return value.capture_routing_metadata_address;
          }
          return 0;
        },
        signature);
    const auto capture_payload_rows_address = std::visit(
        [](const auto &value) -> std::uintptr_t {
          using Signature = std::remove_cvref_t<decltype(value)>;
          if constexpr (std::is_same_v<Signature, MoeExecutionSignatureProjection>) {
            return value.capture_payload_rows_address.value_or(0);
          }
          return 0;
        },
        signature);
    const auto control_graph = std::visit([](const auto &value) { return value.control_graph_address; }, signature);
    auto parameterization = parameterize_graph(embedded_graph, reinterpret_cast<cudaGraph_t>(control_graph), resources,
                                               lane_replacements, capture_routing_metadata_address,
                                               capture_routing_weights_address, capture_payload_rows_address);
    const auto *moe = std::get_if<MoeExecutionSignatureProjection>(&signature);
    const auto has_router = moe != nullptr && moe->routed_expert_count.has_value();
    TORCH_CHECK(parameterization.routing_finalization.has_value() == has_router,
                "xpool Primary Graph Routing Finalization disagrees with Signature");
    if (parameterization.routing_finalization.has_value()) {
      insert_routing_publication(*parameterization.routing_finalization, arena, executor_lane_index,
                                 std::visit([](const auto &value) { return value.payload_row_capacity; }, signature),
                                 device_state);
    }

    host_schemas.push_back(DiscoveredBindingSchema{
        .site_begin = host_sites.size(),
        .site_count = parameterization.binding_sites.size(),
    });
    maximum_site_count = std::max(maximum_site_count, parameterization.binding_sites.size());
    host_sites.insert(host_sites.end(), parameterization.binding_sites.begin(), parameterization.binding_sites.end());
    if (executor_lane_index == 0) {
      xpool::hooks::FfnPrimaryGraphParameterizePostEvent::hooks(
          {.execution_signature_index = signature_index,
           .graph = embedded_graph,
           .binding_site_count = parameterization.binding_sites.size()});
    }
  }
  if (signature_count == 0) {
    auto empty_node = cudaGraphNode_t{};
    C10_CUDA_CHECK(cudaGraphAddEmptyNode(&empty_node, compute_bodies[compute_empty_branch_index], nullptr, 0));
  }
  if (!host_schemas.empty()) {
    const auto bytes = xpool::utils::checked::prod(host_schemas.size(), sizeof(DiscoveredBindingSchema));
    C10_CUDA_CHECK(cudaMalloc(&lane.schemas, bytes));
    C10_CUDA_CHECK(cudaMemcpy(lane.schemas, host_schemas.data(), bytes, cudaMemcpyHostToDevice));
  }
  if (!host_sites.empty()) {
    const auto bytes = xpool::utils::checked::prod(host_sites.size(), sizeof(BindingSite));
    C10_CUDA_CHECK(cudaMalloc(&lane.sites, bytes));
    C10_CUDA_CHECK(cudaMemcpy(lane.sites, host_sites.data(), bytes, cudaMemcpyHostToDevice));
  }
  if (maximum_site_count != 0) {
    C10_CUDA_CHECK(
        cudaMalloc(&lane.updates, xpool::utils::checked::prod(maximum_site_count, sizeof(cudaGraphKernelNodeUpdate))));
  }
  auto *device_schemas = static_cast<DiscoveredBindingSchema *>(lane.schemas);
  auto *device_sites = static_cast<BindingSite *>(lane.sites);
  auto *device_updates = static_cast<cudaGraphKernelNodeUpdate *>(lane.updates);

  // Phase: Acquire - Observe one new LaneExecution and all input or routing
  // publications required by its selected layer and lease.
  void *wait_arguments[] = {
      const_cast<xpool::fabric::ArenaView *>(&arena),
      &executor_lane_index,
      &while_handle,
      &compute_handle,
      &device_state,
      &device_layer_entries,
      &layer_entry_count_,
      &device_capacity_entries,
      const_cast<std::size_t *>(&signature_count),
      const_cast<std::uint32_t *>(&compute_empty_branch_index),
  };
  const auto wait_node = xpool::utils::graph::add_kernel_node(
      while_body, reinterpret_cast<const void *>(wait_pre_compute), dim3{1, 1, 1}, dim3{1, 1, 1}, 0, wait_arguments);

  // Phase: Bind - Patch only the discovered weight sites for the selected
  // layer; capture geometry and Lane-local addresses remain stable.
  void *bind_arguments[] = {
      const_cast<xpool::fabric::ArenaView *>(&arena),
      &executor_lane_index,
      &while_handle,
      &compute_handle,
      &device_state,
      &device_layer_entries,
      &device_schemas,
      const_cast<std::uint32_t *>(&compute_empty_branch_index),
      &device_sites,
      &device_updates,
  };
  const auto bind_node = xpool::utils::graph::add_kernel_node(while_body, reinterpret_cast<const void *>(bind_compute),
                                                              dim3{1, 1, 1}, dim3{1, 1, 1}, 0, bind_arguments);
  xpool::utils::graph::add_dependency(while_body, wait_node, bind_node);

  // Phase: Compute - The bound signature branch produces this TP rank's
  // partial output and, for the router owner, publishes finalized routing data.
  xpool::utils::graph::add_dependency(while_body, bind_node, compute_node);

  auto delivery_handle = xpool::utils::graph::create_conditional_handle(while_body);
  const auto delivery_node = xpool::utils::graph::add_switch_node(while_body, delivery_handle, kDeliveryBranchCount);
  const auto delivery_bodies = xpool::utils::graph::conditional_bodies(delivery_node);

  auto peer_partial_handle = xpool::utils::graph::create_conditional_handle(lane.graph, 0, cudaGraphCondAssignDefault);

  // Phase: Prepare Delivery - Select direct or complete delivery and publish
  // PartialReady only after the local partial payload is visible to its readers.
  void *prepare_arguments[] = {
      const_cast<xpool::fabric::ArenaView *>(&arena),
      &executor_lane_index,
      &while_handle,
      &delivery_handle,
      &peer_partial_handle,
      &device_state,
  };
  const auto prepare_node =
      xpool::utils::graph::add_kernel_node(while_body, reinterpret_cast<const void *>(prepare_delivery), dim3{1, 1, 1},
                                           dim3{kLaneControlBlockSize, 1, 1}, reader_scratch_bytes, prepare_arguments);
  xpool::utils::graph::add_dependency(while_body, compute_node, prepare_node);
  xpool::utils::graph::add_dependency(while_body, prepare_node, delivery_node);

  // Phase: Deliver - Direct delivery forwards each partial; complete delivery
  // waits for peer partials, reduces in PE order, and writes the admitted output.
  void *delivery_arguments[] = {
      const_cast<xpool::fabric::ArenaView *>(&arena),
      &executor_lane_index,
  };
  static_cast<void>(xpool::utils::graph::add_kernel_node(delivery_bodies[kDirectDeliveryBranch],
                                                         reinterpret_cast<const void *>(deliver_direct_partial_output),
                                                         dim3{static_cast<unsigned int>(delivery_block_count), 1, 1},
                                                         dim3{kLaneControlBlockSize, 1, 1}, 0, delivery_arguments));
  const auto complete_delivery_body = delivery_bodies[kCompleteDeliveryBranch];
  // Complete-delivery subprotocol: publish local PartialReady, observe every
  // peer publication, then perform ordered reduction and remote delivery.
  const auto peer_partial_node = xpool::utils::graph::add_while_node(complete_delivery_body, peer_partial_handle);
  const auto peer_partial_body = xpool::utils::graph::conditional_bodies(peer_partial_node).front();
  void *peer_partial_arguments[] = {
      const_cast<xpool::fabric::ArenaView *>(&arena),
      &executor_lane_index,
      &while_handle,
      &peer_partial_handle,
      &device_state,
  };
  static_cast<void>(xpool::utils::graph::add_kernel_node(peer_partial_body,
                                                         reinterpret_cast<const void *>(probe_peer_partials),
                                                         dim3{1, 1, 1}, dim3{1, 1, 1}, 0, peer_partial_arguments));
  const auto reduce_node = xpool::utils::graph::add_kernel_node(
      complete_delivery_body, reinterpret_cast<const void *>(deliver_complete_output_range),
      dim3{static_cast<unsigned int>(delivery_block_count), 1, 1}, dim3{kLaneControlBlockSize, 1, 1}, 0,
      delivery_arguments);
  xpool::utils::graph::add_dependency(complete_delivery_body, peer_partial_node, reduce_node);

  // Phase: Complete - Remote writes are quiet before Completion publication;
  // only then may the Coordinator commit output and eventually reuse the Lane.
  void *completion_arguments[] = {
      const_cast<xpool::fabric::ArenaView *>(&arena),
      &executor_lane_index,
      &while_handle,
      &device_state,
  };
  const auto completion_node =
      xpool::utils::graph::add_kernel_node(while_body, reinterpret_cast<const void *>(complete_lane_execution),
                                           dim3{1, 1, 1}, dim3{kLaneControlBlockSize, 1, 1}, 0, completion_arguments);
  xpool::utils::graph::add_dependency(while_body, delivery_node, completion_node);

  xpool::hooks::FfnLaneGraphBuildPostEvent::hooks({.executor_lane_index = executor_lane_index,
                                                   .graph = lane.graph,
                                                   .compute_switch = compute_node,
                                                   .delivery_switch = delivery_node});

  // Installation uploads every Lane before activate() launches any resident
  // loop, so partial installation never becomes visible to the Fabric.
  auto instantiate_parameters = cudaGraphInstantiateParams{};
  C10_CUDA_CHECK(cudaGraphInstantiateWithParams(&lane.executable, lane.graph, &instantiate_parameters));
  C10_CUDA_CHECK(cudaGraphUpload(lane.executable, lane.stream.get()));
  C10_CUDA_CHECK(cudaStreamSynchronize(lane.stream.get()));
}

void ExecutionRuntime::materialize(const ExecutionProjection &projection, xpool::fabric::ArenaView arena,
                                   const xpool::fabric::ArenaLayout &layout,
                                   const xpool::fabric::ArenaProjection &fabric_projection, std::size_t ffnagent_index,
                                   std::uint32_t *activation_count) {
  TORCH_CHECK(!attempted_, "xpool FfnAgent execution installation cannot be retried");
  attempted_ = true;
  const auto signature_count = projection.signatures.size();
  TORCH_CHECK(signature_count <= std::numeric_limits<std::uint32_t>::max(),
              "xpool execution Signature count exceeds CUDA SWITCH capacity");

  auto device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  try {
    xpool::hooks::FfnExecutionInstallPreEvent::hooks(
        {.fabric_projection = fabric_projection, .ffnagent_index = ffnagent_index});
    validate_layer_resource_targets(projection, device);
    materialize_shared_tables(projection, fabric_projection, ffnagent_index);
    lanes_.reserve(layout.executor_lane_count);
    for (auto executor_lane_index = std::size_t{0}; executor_lane_index < layout.executor_lane_count;
         ++executor_lane_index) {
      materialize_lane(projection, arena, layout, fabric_projection, ffnagent_index, executor_lane_index, device,
                       activation_count);
    }
    installed_ = true;
  } catch (...) {
    destroy_noexcept();
    xpool::hooks::FfnExecutionFinalizePostEvent::hooks({});
    throw;
  }
}

void ExecutionRuntime::install(const ExecutionProjection &projection, xpool::fabric::ArenaView arena,
                               const xpool::fabric::ArenaLayout &layout,
                               const xpool::fabric::ArenaProjection &fabric_projection, std::size_t ffnagent_index,
                               std::uint32_t *activation_count) {
  materialize(projection, arena, layout, fabric_projection, ffnagent_index, activation_count);
}

void ExecutionRuntime::activate() {
  TORCH_CHECK(installed_ && !active_, "xpool FfnAgent execution is not ready for activation");
  for (const auto &lane : lanes_) {
    C10_CUDA_CHECK(cudaGraphLaunch(lane.executable, lane.stream.get()));
  }
  active_ = true;
}

void ExecutionRuntime::check_health() const {
  TORCH_CHECK(installed_ && active_, "xpool FfnAgent execution has not been activated");
  for (auto lane_index = std::size_t{0}; lane_index < lanes_.size(); ++lane_index) {
    const auto &lane = lanes_[lane_index];
    const auto status = cudaStreamQuery(lane.stream.get());
    if (status == cudaErrorNotReady) {
      static_cast<void>(cudaGetLastError());
      continue;
    }
    TORCH_CHECK(status == cudaSuccess, "xpool FfnAgent Lane Graph ", lane_index,
                " failed: ", cudaGetErrorString(status));
    TORCH_CHECK(false, "xpool FfnAgent Lane Graph completed unexpectedly");
  }
}

bool ExecutionRuntime::drain_pending() const {
  TORCH_CHECK(installed_, "xpool FfnAgent execution drain requires installation");
  if (!active_) {
    return false;
  }
  auto pending = false;
  for (const auto &lane : lanes_) {
    const auto status = cudaStreamQuery(lane.stream.get());
    if (status == cudaErrorNotReady) {
      static_cast<void>(cudaGetLastError());
      pending = true;
      continue;
    }
    C10_CUDA_CHECK(status);
  }
  return pending;
}

void ExecutionRuntime::finalize() {
  TORCH_CHECK(installed_, "xpool FfnAgent execution finalize requires installation");
  if (active_) {
    TORCH_CHECK(!drain_pending(), "xpool FfnAgent execution finalize requires completed drain");
  }
  for (auto &lane : lanes_) {
    lane.destroy();
  }
  lanes_.clear();
  if (layer_entries_ != nullptr) {
    C10_CUDA_CHECK(cudaFree(layer_entries_));
    layer_entries_ = nullptr;
  }
  if (capacity_entries_ != nullptr) {
    C10_CUDA_CHECK(cudaFree(capacity_entries_));
    capacity_entries_ = nullptr;
  }
  layer_entry_count_ = 0;
  xpool::hooks::FfnExecutionFinalizePostEvent::hooks({});
  active_ = false;
  installed_ = false;
}

std::size_t execution_state_allocation_bytes(std::size_t local_layer_count, std::size_t local_capacity_count,
                                             std::size_t local_signature_count, std::size_t executor_lane_count) {
  const auto layer_bytes = xpool::utils::checked::prod(local_layer_count, sizeof(LayerExecutionEntry));
  const auto capacity_bytes = xpool::utils::checked::prod(local_capacity_count, sizeof(CapacityExecutionEntry));
  const auto schema_bytes =
      xpool::utils::checked::prod(executor_lane_count, local_signature_count, sizeof(DiscoveredBindingSchema));
  const auto lane_state_bytes = xpool::utils::checked::prod(executor_lane_count, sizeof(LaneRuntimeState));
  return xpool::utils::checked::sum(layer_bytes, capacity_bytes, schema_bytes, lane_state_bytes);
}

void ExecutionRuntime::destroy_noexcept() noexcept {
  for (auto &lane : lanes_) {
    try {
      lane.destroy();
    } catch (...) {
    }
  }
  lanes_.clear();
  if (layer_entries_ != nullptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(layer_entries_));
    layer_entries_ = nullptr;
  }
  if (capacity_entries_ != nullptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(capacity_entries_));
    capacity_entries_ = nullptr;
  }
  layer_entry_count_ = 0;
  active_ = false;
  installed_ = false;
}

} // namespace xpool::ffnagent
