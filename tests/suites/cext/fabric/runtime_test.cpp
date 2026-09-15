#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <xpool/fabric/protocol.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/ffn.hpp>
#include <xpool/ffnagent/runtime.cuh>

namespace {

xpool::fabric::ArenaProjection valid_projection() {
  return xpool::fabric::ArenaProjection{
      .generation_high = 1,
      .generation_low = 2,
      .uid = xpool::fabric::Uid::decode(std::string(sizeof(nvshmemx_uniqueid_t) * 2, 'a')),
      .atnagent_count = 2,
      .ffnagent_count = 2,
      .executor_lane_count = 2,
      .scheduler = xpool::fabric::SchedulerPolicy::fifo(),
      .instances =
          {
              xpool::fabric::InstanceProjection{
                  .decode_payload_row_capacity = 4,
                  .prefill_payload_row_capacity = 16,
                  .payload_dtype = c10::ScalarType::BFloat16,
                  .hidden_size = 2048,
                  .group_sum_complete_admitted = false,
                  .atn_tp_size = 2,
                  .atn_dp_size = 1,
                  .atnagent_indices = {0, 1},
                  .layers =
                      {
                          xpool::fabric::InstanceLayerProjection{
                              .layer_id = 0,
                              .kind = xpool::ffn::LayerKind::Dense,
                              .effective_topk = 0,
                              .ffnagent_indices = {0, 1},
                          },
                          xpool::fabric::InstanceLayerProjection{
                              .layer_id = 1,
                              .kind = xpool::ffn::LayerKind::Moe,
                              .effective_topk = 8,
                              .ffnagent_indices = {0, 1},
                          },
                      },
              },
          },
  };
}

xpool::ffnagent::ExecutionProjection
moe_execution_projection(bool router_owner, std::optional<std::uintptr_t> capture_payload_rows_address) {
  const auto routed_expert_count = router_owner ? std::optional<std::size_t>{4} : std::nullopt;
  const auto primary_router = router_owner ? std::optional{xpool::ffnagent::MoeRouterBindingResourceProjection{
                                                 .weight_address = 10,
                                                 .correction_bias_address = std::nullopt,
                                             }}
                                           : std::nullopt;
  const auto control_router = router_owner ? std::optional{xpool::ffnagent::MoeRouterBindingResourceProjection{
                                                 .weight_address = 13,
                                                 .correction_bias_address = std::nullopt,
                                             }}
                                           : std::nullopt;
  const auto target_router = router_owner ? std::optional{xpool::ffnagent::MoeRouterBindingResourceProjection{
                                                .weight_address = 16,
                                                .correction_bias_address = std::nullopt,
                                            }}
                                          : std::nullopt;
  auto signature = xpool::ffnagent::MoeExecutionSignatureProjection{
      .payload_dtype = c10::ScalarType::BFloat16,
      .payload_row_capacity = 1,
      .hidden_size = 2,
      .local_intermediate_size = 2,
      .expert_count = 4,
      .effective_topk = 2,
      .routed_expert_count = routed_expert_count,
      .primary_graph_address = 1,
      .control_graph_address = 2,
      .capture_input_address = 3,
      .capture_partial_address = 4,
      .capture_workspace_address = 5,
      .compute_workspace_bytes = 1,
      .capture_routing_metadata_address = 6,
      .capture_payload_rows_address = capture_payload_rows_address,
      .primary_capture_resources =
          {
              .expert_gate_up_weight_address = 8,
              .expert_down_weight_address = 9,
              .router = primary_router,
          },
      .control_capture_resources =
          {
              .expert_gate_up_weight_address = 11,
              .expert_down_weight_address = 12,
              .router = control_router,
          },
  };
  return {
      .signatures = {signature},
      .layers =
          {
              xpool::ffnagent::LayerExecutionProjection{
                  .instance_index = 0,
                  .layer_ordinal = 0,
                  .execution_signature_indices = {0},
                  .layer_resource_targets =
                      xpool::ffnagent::MoeBindingResourceProjection{
                          .expert_gate_up_weight_address = 14,
                          .expert_down_weight_address = 15,
                          .router = target_router,
                      },
              },
          },
  };
}

} // namespace

TEST(FabricArenaProjectionTest, DerivesTopologyAndComparesExactInputs) {
  const auto projection = valid_projection();

  EXPECT_NO_THROW(projection.validate());
  EXPECT_EQ(projection.pe_count(), 4);
  EXPECT_EQ(projection.coordinator_pe(), 2);
  EXPECT_EQ(projection, valid_projection());

  auto changed_rows = projection;
  ++changed_rows.instances[0].decode_payload_row_capacity;
  EXPECT_NE(changed_rows, projection);
}

TEST(FfnSchedulerPolicyTest, ConstructsTypedVariants) {
  EXPECT_EQ(xpool::fabric::SchedulerPolicy::fifo().type(), xpool::fabric::SchedulerPolicy::Fifo);
  EXPECT_EQ(xpool::fabric::SchedulerPolicy::random(7).type(), xpool::fabric::SchedulerPolicy::Random);
  EXPECT_EQ(xpool::fabric::SchedulerPolicy::fifo(), xpool::fabric::SchedulerPolicy::fifo());
  EXPECT_NE(xpool::fabric::SchedulerPolicy::random(7), xpool::fabric::SchedulerPolicy::random(8));
}

TEST(FfnAgentMemoryGeometryTest, ReportsOwnerAllocationBytes) {
  EXPECT_EQ(xpool::fabric::ffnagent_control_allocation_bytes(false, 2), sizeof(std::uint32_t));
  EXPECT_GT(xpool::fabric::ffnagent_control_allocation_bytes(true, 2), sizeof(std::uint32_t));
  EXPECT_EQ(xpool::ffnagent::execution_state_allocation_bytes(1, 1, 1, 1),
            sizeof(xpool::ffnagent::LayerExecutionEntry) + sizeof(xpool::ffnagent::CapacityExecutionEntry) +
                sizeof(xpool::ffnagent::DiscoveredBindingSchema) + sizeof(xpool::ffnagent::LaneRuntimeState));
}

TEST(FfnExecutionProjectionTest, CorrelatesPayloadRowsSourceWithRouterOwnership) {
  EXPECT_NO_THROW(moe_execution_projection(true, 7).validate());
  EXPECT_NO_THROW(moe_execution_projection(false, std::nullopt).validate());
  EXPECT_THROW(moe_execution_projection(true, std::nullopt).validate(), c10::Error);
  EXPECT_THROW(moe_execution_projection(false, 7).validate(), c10::Error);
}

TEST(FfnExecutionProjectionTest, AcceptsIncreasingNonPowerOfTwoCapacities) {
  auto projection = moe_execution_projection(true, 7);
  auto middle = std::get<xpool::ffnagent::MoeExecutionSignatureProjection>(projection.signatures.front());
  middle.payload_row_capacity = 3;
  auto final = middle;
  final.payload_row_capacity = 16;
  projection.signatures.emplace_back(std::move(middle));
  projection.signatures.emplace_back(std::move(final));
  projection.layers.front().execution_signature_indices = {0, 1, 2};

  EXPECT_NO_THROW(projection.validate());
}

TEST(FfnExecutionProjectionTest, RejectsLayerResourceSchemaMismatch) {
  auto kind_mismatch = moe_execution_projection(true, 7);
  kind_mismatch.layers.front().layer_resource_targets = xpool::ffnagent::DenseBindingResourceProjection{
      .gate_up_weight_address = 17,
      .down_weight_address = 18,
  };
  EXPECT_THROW(kind_mismatch.validate(), c10::Error);

  auto bias_mismatch = moe_execution_projection(true, 7);
  std::get<xpool::ffnagent::MoeBindingResourceProjection>(bias_mismatch.layers.front().layer_resource_targets)
      .router->correction_bias_address = 17;
  EXPECT_THROW(bias_mismatch.validate(), c10::Error);
}

TEST(FfnExecutionProjectionTest, RejectsCrossCapacityWeightGeometryMismatch) {
  auto projection = moe_execution_projection(true, 7);
  auto second = std::get<xpool::ffnagent::MoeExecutionSignatureProjection>(projection.signatures.front());
  second.payload_row_capacity = 2;
  ++second.hidden_size;
  projection.signatures.emplace_back(std::move(second));
  projection.layers.front().execution_signature_indices = {0, 1};

  EXPECT_THROW(projection.validate(), c10::Error);
}

TEST(FabricArenaProjectionTest, RejectsInvalidSemanticInputs) {
  auto projection = valid_projection();
  projection.generation_high = 0;
  projection.generation_low = 0;
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.executor_lane_count = 0;
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.instances.clear();
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.instances[0].decode_payload_row_capacity = 0;
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.instances[0].hidden_size = 0;
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.instances[0].atnagent_indices = {0};
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.instances[0].layers.clear();
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.instances[0].layers[1].layer_id = projection.instances[0].layers[0].layer_id;
  EXPECT_THROW(projection.validate(), c10::Error);

  projection = valid_projection();
  projection.instances[0].layers[1].ffnagent_indices = {0};
  EXPECT_THROW(projection.validate(), c10::Error);
}

TEST(FabricUidTest, EncodesDecodesAndOrdersOpaqueBytes) {
  const auto lower = xpool::fabric::Uid::decode(std::string(sizeof(nvshmemx_uniqueid_t) * 2, '0'));
  const auto higher = xpool::fabric::Uid::decode(std::string(sizeof(nvshmemx_uniqueid_t) * 2, '1'));

  EXPECT_EQ(lower.encode(), std::string(sizeof(nvshmemx_uniqueid_t) * 2, '0'));
  EXPECT_EQ(lower, xpool::fabric::Uid::decode(lower.encode()));
  EXPECT_LT(lower, higher);
  EXPECT_THROW(xpool::fabric::Uid::decode("ab"), c10::Error);
  EXPECT_THROW(xpool::fabric::Uid::decode(std::string(sizeof(nvshmemx_uniqueid_t) * 2, 'A')), c10::Error);
}
