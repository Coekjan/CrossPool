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

template <PublicationRecord Record>
XPOOL_DEVICE_FN inline bool Publication<Record>::test_at_least(std::uint64_t expected_sequence) const {
  xpool::abort_if(expected_sequence == 0);
  return nvshmem_uint64_test(const_cast<std::uint64_t *>(&sequence), NVSHMEM_CMP_GE, expected_sequence) != 0;
}

template <PublicationRecord Record>
XPOOL_DEVICE_FN inline xpool::ffn::ResultCode
Publication<Record>::validate_expected(std::uint64_t expected_sequence, const InvocationKey &expected_key) const {
  return expected_sequence != 0 && record.publication_sequence() == expected_sequence &&
                 record.validate() == xpool::ffn::ResultCode::Ok && record.key == expected_key
             ? xpool::ffn::ResultCode::Ok
             : xpool::ffn::ResultCode::ProtocolMismatch;
}

template <PublicationRecord Record>
XPOOL_DEVICE_FN inline void Publication<Record>::publish_record(const cooperative_groups::thread_block &group,
                                                                int destination_pe) {
  const auto value = record.publication_sequence();
  xpool::abort_if(destination_pe < 0 || value == 0 || record.validate() != xpool::ffn::ResultCode::Ok);
  group.sync();
  nvshmemx_putmem_signal_block(&record, &record, sizeof(record), &sequence, value, NVSHMEM_SIGNAL_SET, destination_pe);
}

template <PublicationRecord Record>
XPOOL_DEVICE_FN inline void Publication<Record>::publish_payload(const cooperative_groups::thread_block &group,
                                                                 int destination_pe,
                                                                 cuda::std::span<std::uint8_t> symmetric_payload) {
  xpool::abort_if(destination_pe < 0 || symmetric_payload.empty());
  group.sync();
  // The payload has no independent publication sequence. Complete its remote
  // write before publishing the record that makes those bytes consumable.
  nvshmemx_putmem_block(symmetric_payload.data(), symmetric_payload.data(), symmetric_payload.size(), destination_pe);
  group.sync();
  if (group.thread_rank() == 0) {
    nvshmem_fence();
  }
  group.sync();
  publish_record(group, destination_pe);
}

template <PublicationRecord Record>
XPOOL_DEVICE_FN inline void
Publication<Record>::publish_after_local_payload(const cooperative_groups::thread_block &group,
                                                 cuda::std::span<const int> destination_pes) {
  group.sync();
  if (destination_pes.empty()) {
    return;
  }
  // Prior nonblocking puts may have been issued by another graph node. Quiet
  // them before any destination can observe the corresponding record.
  if (group.thread_rank() == 0) {
    nvshmem_quiet();
  }
  group.sync();
  for (const auto destination_pe : destination_pes) {
    publish_record(group, destination_pe);
  }
}

template <PublicationRecord Record>
XPOOL_DEVICE_FN inline void
Publication<Record>::publish_after_remote_payload(const cooperative_groups::thread_block &group, int destination_pe) {
  group.sync();
  if (group.thread_rank() == 0) {
    nvshmem_quiet();
  }
  group.sync();
  publish_record(group, destination_pe);
}

XPOOL_DEVICE_FN XPOOL_DEVICE_FORCEINLINE bool Failure::published() const {
  const auto value = nvshmem_signal_fetch(const_cast<std::uint64_t *>(&publication));
  xpool::abort_if(value > 1);
  return value == 1;
}

XPOOL_DEVICE_FN inline bool Failure::try_publish(int coordinator_pe, xpool::ffn::ResultCode result_code,
                                                 const InvocationKey &key, std::size_t layer_ordinal) {
  xpool::abort_if(
      coordinator_pe < 0 || !key.valid() ||
      (result_code != xpool::ffn::ResultCode::ProtocolMismatch && result_code != xpool::ffn::ResultCode::Timeout));
  // Every PE races for one Coordinator-owned claim; only the winner writes the
  // canonical payload that the Coordinator later broadcasts.
  const auto previous = nvshmem_uint64_atomic_compare_swap(&claim, 0, 1, coordinator_pe);
  if (previous != 0) {
    return false;
  }
  payload = FailurePayload{
      .result_code = result_code,
      .origin_pe = nvshmem_my_pe(),
      .key = key,
      .layer_ordinal = layer_ordinal,
  };
  nvshmem_putmem_signal(&payload, &payload, sizeof(payload), &publication, 1, NVSHMEM_SIGNAL_SET, coordinator_pe);
  return true;
}

XPOOL_DEVICE_FN inline void Failure::publish_to_all(int pe_count) const {
  xpool::abort_if(nvshmem_my_pe() < 0 || pe_count <= 0 || nvshmem_my_pe() >= pe_count || !published());
  for (auto destination_pe = 0; destination_pe < pe_count; ++destination_pe) {
    if (destination_pe == nvshmem_my_pe()) {
      continue;
    }
    nvshmem_putmem_signal(const_cast<FailurePayload *>(&payload), &payload, sizeof(payload),
                          const_cast<std::uint64_t *>(&publication), 1, NVSHMEM_SIGNAL_SET, destination_pe);
  }
}

} // namespace xpool::fabric
