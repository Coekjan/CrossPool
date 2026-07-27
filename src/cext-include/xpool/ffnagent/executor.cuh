#pragma once

/// \file xpool/ffnagent/executor.cuh
/// \brief Typed FFNAgent device compute boundary.

#include <cstdint>

#include <cooperative_groups.h>

#include <xpool/abi.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/macros.hpp>

/// FfnAgent-owned Invocation execution operations.
namespace xpool::ffnagent::executor {

/// Execute one admitted distributed FFN invocation through the current body.
///
/// Before Phase 8 this extension boundary implements only FfnAgent debug
/// loopback. It does not load or execute Dense or MoE weights.
/// \param group Complete cooperative block executing the Invocation.
/// \param invocation Validated dynamic invocation facts.
/// \param model Validated static model geometry.
/// \param layer Validated static layer identity and implementation kind.
/// \param input Executor-local input payload.
/// \param output Executor-local output payload.
/// \return Ok after writing a valid output, ProtocolMismatch for invalid
/// invocation or layer facts, or NotImplemented for unsupported execution
/// policy or geometry.
/// \pre Every thread in the cooperative block calls with identical arguments.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode execute(
    const cooperative_groups::thread_block &group,
    const xpool::fabric::FfnInvocation &invocation,
    const xpool::fabric::FabricModelLayout &model,
    const xpool::fabric::FabricLayerLayout &layer,
    const void *input,
    void *output);

} // namespace xpool::ffnagent::executor
