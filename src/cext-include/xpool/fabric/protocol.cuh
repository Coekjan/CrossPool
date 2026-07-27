#pragma once

/// \file xpool/fabric/protocol.cuh
/// \brief NVSHMEM publication and canonical-failure device mechanics.

#include <cstddef>
#include <cstdint>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/macros.hpp>

namespace xpool::fabric {

template <FabricRecord Record>
XPOOL_DEVICE_FN inline std::uint64_t FabricPublication<Record>::observe() const {
  return nvshmem_signal_fetch(const_cast<std::uint64_t *>(&sequence));
}

template <FabricRecord Record>
XPOOL_DEVICE_FN inline void FabricPublication<Record>::publish(
    const cooperative_groups::thread_block_tile<32> &, std::uint64_t value, int destination_pe) {
  xpool::abort_if(value == 0 || destination_pe < 0 || record.key.invocation_sequence != value ||
                  record.validate() != xpool::abi::FfnResultCode::Ok);
  nvshmemx_putmem_signal_warp(&record, &record, sizeof(record), &sequence, value, NVSHMEM_SIGNAL_SET, destination_pe);
}

template <FabricRecord Record>
XPOOL_DEVICE_FN inline void FabricPublication<Record>::publish(
    const cooperative_groups::thread_block_tile<32> &group, std::uint64_t value, int destination_pe, void *payload,
    std::size_t payload_bytes) {
  xpool::abort_if(value == 0 || destination_pe < 0 || payload == nullptr || payload_bytes == 0 ||
                  record.key.invocation_sequence != value ||
                  record.validate() != xpool::abi::FfnResultCode::Ok);
  nvshmemx_putmem_warp(payload, payload, payload_bytes, destination_pe);
  group.sync();
  if (group.thread_rank() == 0) {
    nvshmem_fence();
  }
  group.sync();
  nvshmemx_putmem_signal_warp(&record, &record, sizeof(record), &sequence, value, NVSHMEM_SIGNAL_SET, destination_pe);
}

template <FabricRecord Record>
XPOOL_DEVICE_FN inline void FabricPublication<Record>::publish(const cooperative_groups::thread_block &,
                                                                std::uint64_t value, int destination_pe) {
  xpool::abort_if(value == 0 || destination_pe < 0 || record.key.invocation_sequence != value ||
                  record.validate() != xpool::abi::FfnResultCode::Ok);
  nvshmemx_putmem_signal_block(&record, &record, sizeof(record), &sequence, value, NVSHMEM_SIGNAL_SET, destination_pe);
}

template <FabricRecord Record>
XPOOL_DEVICE_FN inline void FabricPublication<Record>::publish(const cooperative_groups::thread_block &group,
                                                                std::uint64_t value, int destination_pe, void *payload,
                                                                std::size_t payload_bytes) {
  xpool::abort_if(value == 0 || destination_pe < 0 || payload == nullptr || payload_bytes == 0 ||
                  record.key.invocation_sequence != value ||
                  record.validate() != xpool::abi::FfnResultCode::Ok);
  nvshmemx_putmem_block(payload, payload, payload_bytes, destination_pe);
  group.sync();
  if (group.thread_rank() == 0) {
    nvshmem_fence();
  }
  group.sync();
  nvshmemx_putmem_signal_block(&record, &record, sizeof(record), &sequence, value, NVSHMEM_SIGNAL_SET, destination_pe);
}

template <FabricRecord Record> XPOOL_DEVICE_FN inline void FabricPublication<Record>::clear() {
  nvshmemx_signal_op(&sequence, 0, NVSHMEM_SIGNAL_SET, nvshmem_my_pe());
  nvshmem_quiet();
}

XPOOL_DEVICE_FN inline bool FabricFailure::published() const {
  const auto value = nvshmem_signal_fetch(const_cast<std::uint64_t *>(&publication));
  xpool::abort_if(value > 1);
  return value == 1;
}

XPOOL_DEVICE_FN inline bool FabricFailure::try_publish(int coordinator_pe, xpool::abi::FfnResultCode result_code,
                                                       const FfnInvocationKey &key, std::size_t layer_ordinal) {
  xpool::abort_if(coordinator_pe < 0 || !key.valid() ||
                  (result_code != xpool::abi::FfnResultCode::ProtocolMismatch &&
                   result_code != xpool::abi::FfnResultCode::NotImplemented));
  const auto previous = nvshmem_uint64_atomic_compare_swap(&claim, 0, 1, coordinator_pe);
  if (previous != 0) {
    return false;
  }
  payload = FabricFailurePayload{
      .result_code = result_code.value(),
      .origin_pe = nvshmem_my_pe(),
      .key = key,
      .layer_ordinal = layer_ordinal,
  };
  nvshmem_putmem_signal(&payload, &payload, sizeof(payload), &publication, 1, NVSHMEM_SIGNAL_SET, coordinator_pe);
  return true;
}

XPOOL_DEVICE_FN inline void FabricFailure::publish_to_all(int pe_count) const {
  xpool::abort_if(nvshmem_my_pe() < 0 || pe_count <= 0 || nvshmem_my_pe() >= pe_count || !published());
  for (auto destination_pe = 0; destination_pe < pe_count; ++destination_pe) {
    if (destination_pe == nvshmem_my_pe()) {
      continue;
    }
    nvshmem_putmem_signal(const_cast<FabricFailurePayload *>(&payload), &payload, sizeof(payload),
                          const_cast<std::uint64_t *>(&publication), 1, NVSHMEM_SIGNAL_SET, destination_pe);
  }
}

} // namespace xpool::fabric
