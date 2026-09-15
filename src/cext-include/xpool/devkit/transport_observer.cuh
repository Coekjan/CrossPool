#pragma once

/// \file xpool/devkit/transport_observer.cuh
/// \brief Device implementations for Transport Observer state transitions.

#include <xpool/devkit/transport_observer.hpp>
#include <xpool/utils/time.cuh>

namespace xpool::devkit::transport_observer {

XPOOL_DEVICE_FN inline void Record::begin_instance(std::uint64_t trace_id, std::size_t payload_rows,
                                                   const xpool::transport::RequestMetadata &request) {
  xpool::abort_if(trace_id == 0 || payload_rows == 0 || !request.valid());
  trace_id_ = trace_id;
  payload_rows_ = payload_rows;
  request_ = request;
  result_code_ = xpool::ffn::ResultCode::ProtocolMismatch;
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::RequestStagingStarted>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::begin_atnagent(std::uint64_t trace_id, std::size_t payload_rows,
                                                   const xpool::transport::RequestMetadata &request) {
  xpool::abort_if(trace_id == 0);
  trace_id_ = trace_id;
  payload_rows_ = payload_rows;
  request_ = request;
  result_code_ = xpool::ffn::ResultCode::ProtocolMismatch;
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::RequestObserved>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::request_staging_completed() {
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::RequestStagingCompleted,
                            xpool::hooks::TransportProtocolEventKind::RequestStagingStarted>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::request_published() {
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::RequestPublished,
                            xpool::hooks::TransportProtocolEventKind::RequestStagingCompleted>(
      xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::execution_started() {
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::ExecutionStarted,
                            xpool::hooks::TransportProtocolEventKind::RequestObserved>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::execution_completed() {
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::ExecutionCompleted,
                            xpool::hooks::TransportProtocolEventKind::ExecutionStarted>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::result_published(xpool::ffn::ResultCode result_code) {
  if (recorded(xpool::hooks::TransportProtocolEventKind::ExecutionStarted)) {
    xpool::abort_if(!recorded(xpool::hooks::TransportProtocolEventKind::ExecutionCompleted));
  }
  result_code_ = result_code;
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::ResultPublished,
                            xpool::hooks::TransportProtocolEventKind::RequestObserved>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::result_observed(xpool::ffn::ResultCode effective_result_code) {
  result_code_ = effective_result_code;
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::ResultObserved,
                            xpool::hooks::TransportProtocolEventKind::RequestPublished>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::output_copied() {
  xpool::abort_if(result_code_ != xpool::ffn::ResultCode::Ok);
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::OutputCopied,
                            xpool::hooks::TransportProtocolEventKind::ResultObserved>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::result_acknowledged() {
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::ResultAcknowledged,
                            xpool::hooks::TransportProtocolEventKind::OutputCopied>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::closed(xpool::ffn::ResultCode result_code) {
  xpool::abort_if(recorded(xpool::hooks::TransportProtocolEventKind::RequestPublished));
  result_code_ = result_code;
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::Closed,
                            xpool::hooks::TransportProtocolEventKind::RequestStagingStarted>(xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::closed() {
  xpool::abort_if(recorded(xpool::hooks::TransportProtocolEventKind::ResultAcknowledged));
  timeline_.template record<xpool::hooks::TransportProtocolEventKind::Closed,
                            xpool::hooks::TransportProtocolEventKind::ResultObserved>(xpool::utils::time::now());
}

} // namespace xpool::devkit::transport_observer
