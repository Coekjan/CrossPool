#include <xpool/fabric/projection.hpp>

#include <limits>
#include <unordered_set>

#include <c10/util/Exception.h>

#include <xpool/utils/checked.hpp>

namespace xpool::fabric {

void ArenaProjection::validate() const {
  TORCH_CHECK(atnagent_count != 0 && ffnagent_count != 0, "xpool Fabric topology requires positive Agent counts");
  const auto participant_count = xpool::utils::checked::sum(atnagent_count, ffnagent_count);
  TORCH_CHECK(participant_count <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "xpool Fabric PE count exceeds the NVSHMEM integer domain");
  TORCH_CHECK(generation_high != 0 || generation_low != 0, "xpool Fabric generation must be nonzero");
  TORCH_CHECK(executor_lane_count != 0, "xpool Fabric join requires at least one Executor Lane");
  TORCH_CHECK(!instances.empty(), "xpool Fabric join requires at least one Instance");

  for (const auto &instance : instances) {
    TORCH_CHECK(instance.decode_payload_row_capacity != 0 && instance.prefill_payload_row_capacity != 0 &&
                    instance.hidden_size != 0 && instance.atn_tp_size != 0 && instance.atn_dp_size != 0,
                "xpool Fabric Instance Projection has zero geometry");
    TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(instance.payload_dtype),
                "xpool Fabric Instance Projection requires BF16 or FP16 payloads");
    TORCH_CHECK(!(instance.atn_tp_size > 1 && instance.atn_dp_size > 1),
                "xpool Fabric does not support combined attention TP-by-DP");
    TORCH_CHECK(xpool::utils::checked::prod(instance.atn_tp_size, instance.atn_dp_size) ==
                    instance.atnagent_indices.size(),
                "xpool Fabric Instance topology membership has the wrong width");
    auto atnagent_indices = std::unordered_set<std::size_t>{};
    for (const auto index : instance.atnagent_indices) {
      TORCH_CHECK(index < atnagent_count && atnagent_indices.insert(index).second,
                  "xpool Fabric Instance contains an invalid AtnAgent index");
    }
    TORCH_CHECK(!instance.layers.empty(), "xpool Fabric Instance requires at least one FFN layer");
    auto layer_ids = std::unordered_set<std::size_t>{};
    layer_ids.reserve(instance.layers.size());
    auto tp_size = std::size_t{0};
    for (const auto &layer : instance.layers) {
      TORCH_CHECK(xpool::ffn::is_valid(layer.kind), "xpool Fabric Instance contains an invalid FFN layer kind");
      TORCH_CHECK(layer_ids.insert(layer.layer_id).second, "xpool Fabric Instance contains duplicate FFN layer ids");
      TORCH_CHECK(!layer.ffnagent_indices.empty(), "xpool Fabric layer requires an FFN execution group");
      if (tp_size == 0) {
        tp_size = layer.ffnagent_indices.size();
      }
      TORCH_CHECK(layer.ffnagent_indices.size() == tp_size, "xpool Fabric Instance layers disagree on FFN TP width");
      auto ffnagent_indices = std::unordered_set<std::size_t>{};
      for (const auto index : layer.ffnagent_indices) {
        TORCH_CHECK(index < ffnagent_count && ffnagent_indices.insert(index).second,
                    "xpool Fabric layer contains an invalid FfnAgent index");
      }
      TORCH_CHECK((layer.kind == xpool::ffn::LayerKind::Dense && layer.effective_topk == 0) ||
                      (layer.kind == xpool::ffn::LayerKind::Moe && layer.effective_topk != 0),
                  "xpool Fabric layer has invalid effective TopK geometry");
    }
  }
}

} // namespace xpool::fabric
