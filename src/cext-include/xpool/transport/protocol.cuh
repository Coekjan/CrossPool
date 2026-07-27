#pragma once

/// \file xpool/transport/protocol.cuh
/// \brief System-scope atomic transitions for the Transport mailbox.

#include <cuda/atomic>

#include <xpool/abort.hpp>
#include <xpool/atomic.cuh>
#include <xpool/transport/protocol.hpp>

namespace xpool::transport {

XPOOL_DEVICE_FN inline MailboxStatus TransportMailbox::observe() {
  const auto value = xpool::atomic::system_atomic(status).load(cuda::memory_order_acquire);
  xpool::abort_if(value > static_cast<std::uint32_t>(MailboxStatus::Closed));
  return static_cast<MailboxStatus>(value);
}

XPOOL_DEVICE_FN inline void TransportMailbox::open() {
  xpool::abort_if(observe() != MailboxStatus::Dormant);
  xpool::atomic::store_release(status, static_cast<std::uint32_t>(MailboxStatus::Idle));
}

XPOOL_DEVICE_FN inline bool TransportMailbox::try_begin_staging() {
  const auto observed = xpool::atomic::compare_exchange_acq_rel(
      status, static_cast<std::uint32_t>(MailboxStatus::Idle),
      static_cast<std::uint32_t>(MailboxStatus::Staging));
  xpool::abort_if(observed > static_cast<std::uint32_t>(MailboxStatus::Closed));
  return observed == static_cast<std::uint32_t>(MailboxStatus::Idle);
}

XPOOL_DEVICE_FN inline void TransportMailbox::publish_request() {
  xpool::abort_if(observe() != MailboxStatus::Staging || payload_rows == 0 || !request.valid());
  xpool::atomic::store_release(status, static_cast<std::uint32_t>(MailboxStatus::Published));
}

XPOOL_DEVICE_FN inline void TransportMailbox::publish_result(xpool::abi::FfnResultCode code) {
  xpool::abort_if(observe() != MailboxStatus::Published);
  result_code = code.value();
  xpool::atomic::store_release(status, static_cast<std::uint32_t>(MailboxStatus::Evaluated));
}

XPOOL_DEVICE_FN inline void TransportMailbox::acknowledge() {
  xpool::abort_if(observe() != MailboxStatus::Evaluated);
  result_code = xpool::abi::FfnResultCode::ProtocolMismatch;
  payload_rows = 0;
  request = {};
  xpool::atomic::store_release(status, static_cast<std::uint32_t>(MailboxStatus::Idle));
}

XPOOL_DEVICE_FN inline void TransportMailbox::close_staging() {
  xpool::abort_if(observe() != MailboxStatus::Staging);
  xpool::atomic::store_release(status, static_cast<std::uint32_t>(MailboxStatus::Closed));
}

XPOOL_DEVICE_FN inline void TransportMailbox::close_evaluated() {
  xpool::abort_if(observe() != MailboxStatus::Evaluated);
  xpool::atomic::store_release(status, static_cast<std::uint32_t>(MailboxStatus::Closed));
}

XPOOL_DEVICE_FN inline bool TransportMailbox::try_close_idle() {
  const auto observed = xpool::atomic::compare_exchange_acq_rel(
      status, static_cast<std::uint32_t>(MailboxStatus::Idle),
      static_cast<std::uint32_t>(MailboxStatus::Closed));
  xpool::abort_if(observed > static_cast<std::uint32_t>(MailboxStatus::Closed));
  return observed == static_cast<std::uint32_t>(MailboxStatus::Idle);
}

} // namespace xpool::transport
