#include <xpool/transport/layout.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

#include <c10/util/Exception.h>

#include <xpool/abi.hpp>
#include <xpool/transport/arena.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/layout.hpp>

namespace xpool::transport {

namespace {

struct TransportArenaRegions {
  xpool::utils::layout::LayoutRegion layout;
  xpool::utils::layout::LayoutRegion state;
  xpool::utils::layout::LayoutRegion mailbox;
  xpool::utils::layout::LayoutRegion input_payload;
  xpool::utils::layout::LayoutRegion output_payload;
  xpool::utils::layout::LayoutRegion dp_rank_payload_rows;
  std::size_t total_bytes;

  TransportArenaRegions(std::size_t payload_bytes, std::size_t dp_rank_count) {
    using xpool::utils::layout::LayoutRegionSpec;
    const auto specs = std::to_array<LayoutRegionSpec>({
        LayoutRegionSpec::object<ArenaLayout>("transport layout"),
        LayoutRegionSpec::object<ArenaState>("transport state"),
        LayoutRegionSpec::object<Mailbox>("transport mailbox"),
        LayoutRegionSpec::bytes("transport input payload", payload_bytes, xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::bytes("transport output payload", payload_bytes, xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::array<std::uint32_t>("transport DP-rank payload rows", dp_rank_count),
    });
    const auto plan = xpool::utils::layout::LayoutPlan{specs, xpool::arena::kAllocationAlignment};
    auto index = std::size_t{0};
    layout = plan[index++];
    state = plan[index++];
    mailbox = plan[index++];
    input_payload = plan[index++];
    output_payload = plan[index++];
    dp_rank_payload_rows = plan[index++];
    TORCH_CHECK(index == plan.regions.size(), "xpool transport arena region plan is incomplete");
    total_bytes = plan.total_bytes;
  }
};

} // namespace

ArenaLayout ArenaLayout::create(std::size_t instance_index, std::size_t instance_rank, std::size_t atn_tp_rank,
                                std::size_t atn_tp_size, std::size_t atn_dp_rank, std::size_t atn_dp_size,
                                std::size_t payload_row_capacity, std::size_t hidden_size,
                                c10::ScalarType payload_dtype) {
  TORCH_CHECK(payload_row_capacity != 0, "xpool transport arena requires a positive payload row capacity");
  TORCH_CHECK(hidden_size != 0, "xpool transport arena requires a positive hidden size");
  TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(payload_dtype),
              "xpool transport arena requires BF16 or FP16 payloads");
  TORCH_CHECK(atn_tp_size != 0 && atn_tp_rank < atn_tp_size, "xpool transport arena has invalid attention TP topology");
  TORCH_CHECK(atn_dp_size != 0 && atn_dp_rank < atn_dp_size, "xpool transport arena has invalid attention DP topology");

  const auto payload_row_bytes =
      xpool::utils::checked::prod(hidden_size, static_cast<std::size_t>(c10::elementSize(payload_dtype)));
  const auto bytes = xpool::utils::checked::prod(payload_row_capacity, payload_row_bytes);
  const auto dp_rank_count = atn_dp_size == 1 ? std::size_t{0} : atn_dp_size;
  const TransportArenaRegions regions{bytes, dp_rank_count};
  return ArenaLayout{
      .header = {.magic = kTransportArenaMagic,
                 .abi_version = xpool::abi::kVersion,
                 .layout_size = sizeof(ArenaLayout),
                 .total_bytes = regions.total_bytes,
                 .state_offset = regions.state.offset},
      .instance_index = instance_index,
      .instance_rank = instance_rank,
      .atn_tp_rank = atn_tp_rank,
      .atn_tp_size = atn_tp_size,
      .atn_dp_rank = atn_dp_rank,
      .atn_dp_size = atn_dp_size,
      .payload_row_capacity = payload_row_capacity,
      .hidden_size = hidden_size,
      .payload_row_bytes = payload_row_bytes,
      .payload_dtype = payload_dtype,
      .mailbox_offset = regions.mailbox.offset,
      .input_payload_offset = regions.input_payload.offset,
      .output_payload_offset = regions.output_payload.offset,
      .dp_rank_payload_rows_offset = dp_rank_count == 0 ? 0 : regions.dp_rank_payload_rows.offset,
  };
}

void ArenaLayout::validate() const {
  TORCH_CHECK(header.magic == kTransportArenaMagic, "xpool transport arena header magic does not match");
  TORCH_CHECK(header.abi_version == xpool::abi::kVersion && header.layout_size == sizeof(ArenaLayout),
              "xpool transport arena layout has an incompatible ABI");
  TORCH_CHECK(payload_row_capacity != 0 && hidden_size != 0 && xpool::ffn::is_supported_payload_dtype(payload_dtype),
              "xpool transport arena layout has invalid tensor geometry");
  TORCH_CHECK(atn_tp_size != 0 && atn_tp_rank < atn_tp_size,
              "xpool transport arena layout has invalid attention TP topology");
  TORCH_CHECK(atn_dp_size != 0 && atn_dp_rank < atn_dp_size,
              "xpool transport arena layout has invalid attention DP topology");

  const auto expected_payload_row_bytes =
      xpool::utils::checked::prod(hidden_size, static_cast<std::size_t>(c10::elementSize(payload_dtype)));
  const auto bytes = xpool::utils::checked::prod(payload_row_capacity, expected_payload_row_bytes);
  const auto dp_rank_count = atn_dp_size == 1 ? std::size_t{0} : atn_dp_size;
  const TransportArenaRegions regions{bytes, dp_rank_count};
  const auto expected_dp_offset = dp_rank_count == 0 ? std::size_t{0} : regions.dp_rank_payload_rows.offset;
  TORCH_CHECK(payload_row_bytes == expected_payload_row_bytes && header.total_bytes == regions.total_bytes &&
                  header.state_offset == regions.state.offset && mailbox_offset == regions.mailbox.offset &&
                  input_payload_offset == regions.input_payload.offset &&
                  output_payload_offset == regions.output_payload.offset &&
                  dp_rank_payload_rows_offset == expected_dp_offset,
              "xpool transport arena layout does not match canonical geometry");
}

} // namespace xpool::transport
