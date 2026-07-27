#pragma once

/// \file xpool/transport/trace.cuh
/// \brief Device implementations for semantic trace state transitions.

#include <xpool/transport/trace.hpp>
#include <xpool/utils/time.cuh>

namespace xpool::transport {

XPOOL_DEVICE_FN inline void TransportTraceRecord::begin(std::uint64_t trace_id, std::size_t payload_rows,
                             const FfnRequestMetadata &request) {
    xpool::abort_if(trace_id == 0 || payload_rows == 0 || !request.valid());
    trace_id_ = trace_id;
    payload_rows_ = payload_rows;
    request_ = request;
    result_code_ = xpool::abi::FfnResultCode::ProtocolMismatch;
    timeline_.template record<TransportTraceEvent::StagingStarted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::staging_completed() {
    timeline_.template record<TransportTraceEvent::StagingCompleted,
                              TransportTraceEvent::StagingStarted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::published() {
    timeline_.template record<TransportTraceEvent::Published,
                              TransportTraceEvent::StagingCompleted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::published_observed() {
    timeline_.template record<TransportTraceEvent::PublishedObserved,
                              TransportTraceEvent::Published>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::execution_started() {
    timeline_.template record<TransportTraceEvent::ExecutionStarted,
                              TransportTraceEvent::PublishedObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::execution_admitted() {
    timeline_.template record<TransportTraceEvent::ExecutionAdmitted,
                              TransportTraceEvent::ExecutionStarted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::execution_completed() {
    timeline_.template record<TransportTraceEvent::ExecutionCompleted,
                              TransportTraceEvent::ExecutionStarted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::evaluated(xpool::abi::FfnResultCode result_code) {
    if (result_code == xpool::abi::FfnResultCode::Ok) {
      xpool::abort_if(!recorded(TransportTraceEvent::ExecutionAdmitted) ||
                      !recorded(TransportTraceEvent::ExecutionCompleted));
    } else if (recorded(TransportTraceEvent::ExecutionStarted)) {
      xpool::abort_if(!recorded(TransportTraceEvent::ExecutionCompleted));
    }
    result_code_ = result_code;
    timeline_.template record<TransportTraceEvent::Evaluated,
                              TransportTraceEvent::PublishedObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::evaluated_observed(xpool::abi::FfnResultCode effective_result_code) {
    result_code_ = effective_result_code;
    timeline_.template record<TransportTraceEvent::EvaluatedObserved,
                              TransportTraceEvent::Evaluated>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::output_copied() {
    xpool::abort_if(result_code_ != xpool::abi::FfnResultCode::Ok);
    timeline_.template record<TransportTraceEvent::OutputCopied,
                              TransportTraceEvent::EvaluatedObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::acknowledged() {
    timeline_.template record<TransportTraceEvent::Acknowledged,
                              TransportTraceEvent::OutputCopied>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::closed(xpool::abi::FfnResultCode result_code) {
    xpool::abort_if(recorded(TransportTraceEvent::Published));
    result_code_ = result_code;
    timeline_.template record<TransportTraceEvent::Closed,
                              TransportTraceEvent::StagingStarted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void TransportTraceRecord::closed() {
    xpool::abort_if(recorded(TransportTraceEvent::Acknowledged));
    timeline_.template record<TransportTraceEvent::Closed,
                              TransportTraceEvent::EvaluatedObserved>(xpool::utils::time::now());
  }

} // namespace xpool::transport
