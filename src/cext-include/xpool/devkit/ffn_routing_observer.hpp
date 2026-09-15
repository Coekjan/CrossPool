#pragma once

/// \file xpool/devkit/ffn_routing_observer.hpp
/// \brief Host readout for process-local semantic MoE routing records.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <ATen/core/Tensor.h>

#include <xpool/fabric/protocol.hpp>

namespace xpool::devkit::ffn_routing_observer {

/// One compact Host-owned semantic routing record.
struct Record {
  /// Invocation that produced the routing decision.
  xpool::fabric::InvocationKey key;
  /// Model-local layer ordinal.
  std::size_t layer_ordinal;
  /// CPU int32 tensor shaped as live rows by effective top-k.
  at::Tensor topk_ids;
  /// CPU float32 tensor matching topk_ids.
  at::Tensor topk_weights;
};

/// Repeatable Host-owned readout of one completed observer buffer.
struct Snapshot {
  /// Total number of record slots reserved since installation.
  std::uint64_t sequence;
  /// Number of reservations dropped after capacity was exhausted.
  std::uint64_t dropped;
  /// Retained semantic routing records in reservation order.
  std::vector<Record> records;
};

/// Return the process-local routing snapshot when observation was installed.
/// \pre The related FfnAgent execution is drained or otherwise synchronized.
/// \return A repeatable Host-owned snapshot, or no value outside the installed lifecycle.
/// \throws c10::Error when CUDA copy or retained routing state is invalid.
std::optional<Snapshot> read();

/// Return the process-local Routing Observer allocation size for one record width.
/// \param element_capacity Maximum ID or weight elements per record.
/// \return Required bytes under current Debug Options, or zero when disabled.
/// \throws c10::Error when capacity arithmetic overflows.
std::size_t allocation_bytes(std::size_t element_capacity);

} // namespace xpool::devkit::ffn_routing_observer
