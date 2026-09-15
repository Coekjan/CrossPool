#include <xpool/ffnagent/delivery.cuh>

#include <cstddef>
#include <cstdint>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cooperative_groups.h>
#include <cuda/hierarchy>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/fabric/protocol.cuh>
#include <xpool/macros.hpp>

namespace xpool::ffnagent {

namespace {

struct DeliveryContext {
  const xpool::fabric::LaneExecution &execution;
  const xpool::fabric::InstanceEntry &instance;
  cuda::std::span<const int> ffnagent_pes;
  cuda::std::span<const int> atnagent_pes;
  xpool::fabric::FfnAgentLanePayloadView payload;
  std::size_t tp_rank;
};

XPOOL_DEVICE_FN DeliveryContext resolve_context(xpool::fabric::ArenaView arena, std::size_t executor_lane_index) {
  const auto &layout = arena.layout();
  xpool::abort_if(executor_lane_index >= layout.executor_lane_count);
  const auto &execution = arena.lane_execution_publication(executor_lane_index).record;
  xpool::abort_if(execution.validate() != xpool::ffn::ResultCode::Ok ||
                  execution.key.instance_index >= layout.instance_count);
  const auto &instance = arena.instance_entry(execution.key.instance_index);
  xpool::abort_if(execution.layer_ordinal >= instance.layer_count ||
                  execution.payload_rows > instance.decode_payload_row_capacity &&
                      execution.payload_rows > instance.prefill_payload_row_capacity);
  const auto ffnagent_pes = arena.ffnagent_pes(execution.key.instance_index, execution.layer_ordinal);
  const auto atnagent_pes = arena.atnagent_pes(execution.key.instance_index);
  xpool::abort_if(ffnagent_pes.empty() || ffnagent_pes.size() != instance.ffn_tp_size || atnagent_pes.empty());
  const auto pe = nvshmem_my_pe();
  const auto member = cuda::std::find(ffnagent_pes.begin(), ffnagent_pes.end(), pe);
  xpool::abort_if(member == ffnagent_pes.end());
  return DeliveryContext{
      .execution = execution,
      .instance = instance,
      .ffnagent_pes = ffnagent_pes,
      .atnagent_pes = atnagent_pes,
      .payload = arena.ffnagent_lane_payload(executor_lane_index),
      .tp_rank = static_cast<std::size_t>(member - ffnagent_pes.begin()),
  };
}

template <typename Scalar> XPOOL_DEVICE_FN const Scalar *partial_pointer(const Scalar *local_partial, int source_pe) {
  if (source_pe == nvshmem_my_pe()) {
    return local_partial;
  }
  const auto *pointer = static_cast<const Scalar *>(nvshmem_ptr(local_partial, source_pe));
  xpool::abort_if(pointer == nullptr);
  return pointer;
}

template <typename Scalar>
XPOOL_DEVICE_FN Scalar ordered_sum(cuda::std::span<const Scalar> local_partial, cuda::std::span<const int> ffnagent_pes,
                                   std::size_t index) {
  // PE order is part of the numerical contract: arrival order must not change
  // FP32 accumulation or the single final payload-dtype cast.
  auto sum = 0.0F;
  for (const auto source_pe : ffnagent_pes) {
    sum += static_cast<float>(partial_pointer(local_partial.data(), source_pe)[index]);
  }
  return Scalar{sum};
}

template <typename Scalar>
XPOOL_DEVICE_FN void reduce_tp_partials(const cooperative_groups::thread_block &group, cuda::std::span<Scalar> staging,
                                        cuda::std::span<const Scalar> local_partial,
                                        cuda::std::span<const int> ffnagent_pes) {
  constexpr auto lanes = std::size_t{sizeof(uint4) / sizeof(Scalar)};
  auto vector_begin = std::size_t{0};
  while (vector_begin < local_partial.size() &&
         ((reinterpret_cast<std::uintptr_t>(local_partial.data() + vector_begin) |
           reinterpret_cast<std::uintptr_t>(staging.data() + vector_begin)) &
          (alignof(uint4) - 1)) != 0) {
    ++vector_begin;
  }
  const auto vector_end = vector_begin + ((local_partial.size() - vector_begin) / lanes) * lanes;

  // Scalar edges and the uint4 middle preserve the same per-element PE order;
  // vectorization changes memory traffic, not association order.
  for (auto index = static_cast<std::size_t>(group.thread_rank()); index < vector_begin;
       index += static_cast<std::size_t>(group.size())) {
    staging[index] = ordered_sum(local_partial, ffnagent_pes, index);
  }
  for (auto index = vector_begin + static_cast<std::size_t>(group.thread_rank()) * lanes; index < vector_end;
       index += static_cast<std::size_t>(group.size()) * lanes) {
    auto sums = cuda::std::array<float, lanes>{};
    for (const auto source_pe : ffnagent_pes) {
      const auto *source = partial_pointer(local_partial.data(), source_pe);
      const auto packed = *reinterpret_cast<const uint4 *>(source + index);
      const auto values = cuda::std::bit_cast<cuda::std::array<Scalar, lanes>>(packed);
      for (auto lane = std::size_t{0}; lane < lanes; ++lane) {
        sums[lane] += static_cast<float>(values[lane]);
      }
    }
    auto values = cuda::std::array<Scalar, lanes>{};
    for (auto lane = std::size_t{0}; lane < lanes; ++lane) {
      values[lane] = Scalar{sums[lane]};
    }
    *reinterpret_cast<uint4 *>(staging.data() + index) = cuda::std::bit_cast<uint4>(values);
  }
  for (auto index = vector_end + static_cast<std::size_t>(group.thread_rank()); index < local_partial.size();
       index += static_cast<std::size_t>(group.size())) {
    staging[index] = ordered_sum(local_partial, ffnagent_pes, index);
  }
  group.sync();
}

template <typename Scalar>
XPOOL_DEVICE_FN void deliver_complete_range(const DeliveryContext &context,
                                            const cooperative_groups::thread_block &group, std::size_t begin_element,
                                            std::size_t element_count, cuda::std::span<const int> destination_pes) {
  const auto partial_bytes = context.payload.partial();
  auto staging_bytes = context.payload.complete_output_staging_destination();
  xpool::abort_if(partial_bytes.size() % sizeof(Scalar) != 0 || staging_bytes.size() != partial_bytes.size() ||
                  begin_element > partial_bytes.size() / sizeof(Scalar) ||
                  element_count > partial_bytes.size() / sizeof(Scalar) - begin_element);
  const auto partial =
      cuda::std::span{reinterpret_cast<const Scalar *>(partial_bytes.data()), partial_bytes.size() / sizeof(Scalar)};
  auto staging =
      cuda::std::span{reinterpret_cast<Scalar *>(staging_bytes.data()), staging_bytes.size() / sizeof(Scalar)};

  const auto block_rank = cuda::block.rank(cuda::grid);
  const auto block_count = cuda::block.count(cuda::grid);
  const auto relative_begin = element_count * block_rank / block_count;
  const auto relative_end = element_count * (block_rank + 1) / block_count;
  const auto block_begin = begin_element + relative_begin;
  const auto block_size = relative_end - relative_begin;
  if (block_size == 0) {
    return;
  }

  const auto partial_block = partial.subspan(block_begin, block_size);
  auto delivery = partial_block;
  if (context.ffnagent_pes.size() > 1) {
    auto staging_block = staging.subspan(block_begin, block_size);
    reduce_tp_partials(group, staging_block, partial_block, context.ffnagent_pes);
    delivery = staging_block;
  }

  const auto destination_offset_bytes = block_begin * sizeof(Scalar);
  for (const auto destination_pe : destination_pes) {
    nvshmemx_putmem_nbi_block(context.payload.partial_destination().data() + destination_offset_bytes, delivery.data(),
                              delivery.size_bytes(), destination_pe);
  }
}

} // namespace

XPOOL_KERNEL_FN void deliver_direct_partial_output(xpool::fabric::ArenaView arena, std::size_t executor_lane_index) {
  const auto context = resolve_context(arena, executor_lane_index);
  xpool::abort_if(context.execution.output_requirement != xpool::ffn::OutputRequirement::GroupSumComplete ||
                  !context.instance.group_sum_complete_admitted ||
                  context.ffnagent_pes.size() > context.atnagent_pes.size());
  const auto bytes = context.execution.payload_rows * context.instance.payload_row_bytes;
  const auto block_rank = cuda::block.rank(cuda::grid);
  const auto block_count = cuda::block.count(cuda::grid);
  const auto begin = bytes * block_rank / block_count;
  const auto end = bytes * (block_rank + 1) / block_count;
  if (begin == end) {
    return;
  }
  nvshmemx_putmem_nbi_block(context.payload.partial_destination().data() + begin,
                            context.payload.partial().data() + begin, end - begin,
                            context.atnagent_pes[context.tp_rank]);
}

XPOOL_KERNEL_FN void deliver_complete_output_range(xpool::fabric::ArenaView arena, std::size_t executor_lane_index) {
  if (arena.shutdown_requested() || arena.state().failure.published()) {
    return;
  }
  const auto context = resolve_context(arena, executor_lane_index);
  const auto per_rank_complete = context.execution.output_requirement == xpool::ffn::OutputRequirement::PerRankComplete;
  const auto group_sum_complete =
      context.execution.output_requirement == xpool::ffn::OutputRequirement::GroupSumComplete;
  xpool::abort_if(!per_rank_complete && (!group_sum_complete || !context.instance.group_sum_complete_admitted ||
                                         context.ffnagent_pes.size() <= context.atnagent_pes.size()));

  const auto base = context.execution.payload_rows / context.ffnagent_pes.size();
  const auto extra = context.execution.payload_rows % context.ffnagent_pes.size();
  const auto begin_row = context.tp_rank * base + cuda::std::min(context.tp_rank, extra);
  const auto row_count = base + (context.tp_rank < extra ? 1U : 0U);
  if (row_count == 0) {
    return;
  }
  const auto begin_element = begin_row * context.instance.hidden_size;
  const auto element_count = row_count * context.instance.hidden_size;
  const auto destinations = per_rank_complete ? context.atnagent_pes : context.atnagent_pes.first(1);
  const auto group = cooperative_groups::this_thread_block();
  switch (context.instance.payload_dtype) {
  case c10::ScalarType::Half:
    deliver_complete_range<c10::Half>(context, group, begin_element, element_count, destinations);
    return;
  case c10::ScalarType::BFloat16:
    deliver_complete_range<c10::BFloat16>(context, group, begin_element, element_count, destinations);
    return;
  default:
    xpool::abort();
  }
}

} // namespace xpool::ffnagent
