#pragma once

/// \file xpool/transport/protocol.cuh
/// \brief System-scope atomic transitions for the Transport mailbox.

#include <cuda/atomic>

#include <xpool/abort.hpp>
#include <xpool/transport/protocol.hpp>

namespace xpool::transport {

XPOOL_DEVICE_FN inline MailboxStatus Mailbox::observe() {
  const auto value = cuda::atomic_ref{status}.load(cuda::memory_order_acquire);
  xpool::abort_if(value < MailboxStatus::Dormant || value > MailboxStatus::Closed);
  return value;
}

XPOOL_DEVICE_FN inline void Mailbox::open() {
  xpool::abort_if(observe() != MailboxStatus::Dormant);
  cuda::atomic_ref{status}.store(MailboxStatus::Idle, cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline bool Mailbox::try_begin_staging() {
  auto observed = MailboxStatus::Idle;
  static_cast<void>(cuda::atomic_ref{status}.compare_exchange_strong(
      observed, MailboxStatus::Staging, cuda::memory_order_acq_rel, cuda::memory_order_acquire));
  xpool::abort_if(observed < MailboxStatus::Dormant || observed > MailboxStatus::Closed);
  return observed == MailboxStatus::Idle;
}

XPOOL_DEVICE_FN inline void Mailbox::publish_request() {
  xpool::abort_if(observe() != MailboxStatus::Staging || payload_rows == 0 || !request.valid());
  cuda::atomic_ref{status}.store(MailboxStatus::Published, cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline void Mailbox::publish_result(xpool::ffn::ResultCode code) {
  xpool::abort_if(observe() != MailboxStatus::Published);
  result_code = code;
  cuda::atomic_ref{status}.store(MailboxStatus::Evaluated, cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline void Mailbox::acknowledge() {
  xpool::abort_if(observe() != MailboxStatus::Evaluated);
  result_code = xpool::ffn::ResultCode::ProtocolMismatch;
  payload_rows = 0;
  request = {};
  cuda::atomic_ref{status}.store(MailboxStatus::Idle, cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline void Mailbox::close_staging() {
  xpool::abort_if(observe() != MailboxStatus::Staging);
  cuda::atomic_ref{status}.store(MailboxStatus::Closed, cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline void Mailbox::close_evaluated() {
  xpool::abort_if(observe() != MailboxStatus::Evaluated);
  cuda::atomic_ref{status}.store(MailboxStatus::Closed, cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline bool Mailbox::try_close_idle() {
  auto observed = MailboxStatus::Idle;
  static_cast<void>(cuda::atomic_ref{status}.compare_exchange_strong(
      observed, MailboxStatus::Closed, cuda::memory_order_acq_rel, cuda::memory_order_acquire));
  xpool::abort_if(observed < MailboxStatus::Dormant || observed > MailboxStatus::Closed);
  return observed == MailboxStatus::Idle;
}

} // namespace xpool::transport
