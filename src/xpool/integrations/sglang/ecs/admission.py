"""SGLang-owned selection and commit with matched Elastic Context State capacity decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from http import HTTPStatus
from typing import cast

import torch
import torch.distributed
from sglang.srt.beam_search.batch_tail import append_beam_tail, beam_retraction_order, strip_beam_tail
from sglang.srt.beam_search.fork import free_member_rows
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS, AddReqResult, PrefillAdder, _PrefillAdmission
from sglang.srt.managers.scheduler import _make_abort_req
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import NewTokenRatioTracker
from sglang.srt.mem_cache.allocation import alloc_for_decode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_parallel

from xpool.integrations.sglang.ecs.allocation import SglangAllocation, SglangCapacityMiss
from xpool.integrations.sglang.ecs.capacity import current_capacity_reconciler
from xpool.integrations.sglang.ecs.interfaces import SglangMemoryRequirement, SglangPrefillProposal


class ElasticPrefillAdder(PrefillAdder):
    """Retain engine policy and commits, substituting common logical observations.

    Pure selection runs again for each proposed chunk cap. Temporary prefix
    pins protect the candidate while common cache actions change other nodes.
    The allocation journal owns only new persistent locks and unpublished work.
    """

    tree_cache: UnifiedRadixCache

    @property
    def rem_total_tokens(self) -> float:
        reconciler = current_capacity_reconciler()
        return float(reconciler.state.prefill_budget(self, "total", reconciler.admission_free_units))

    @property
    def cur_rem_tokens(self) -> int:
        reconciler = current_capacity_reconciler()
        return int(reconciler.state.prefill_budget(self, "current", reconciler.admission_free_units))

    @property
    def rem_swa_tokens(self) -> int:
        reconciler = current_capacity_reconciler()
        return int(reconciler.state.prefill_budget(self, "window", reconciler.admission_free_units))

    def proposal_for_admission(
        self, request: Req, admission: _PrefillAdmission, *, continuation: bool = False
    ) -> SglangPrefillProposal:
        return current_capacity_reconciler().state.prefill_proposal(self, request, admission, continuation=continuation)

    def _req_inc_lock_ref(self, req: Req) -> None:
        """Keep the original acquisition and retain its exact rollback receipt."""

        super()._req_inc_lock_ref(req)
        attempt = SglangAllocation.current()
        if attempt is None:
            raise RuntimeError("xpool Prefill commit requires an allocation attempt")
        attempt.record_lock(req.lock_receipt)

    def coordinate_prefill(
        self,
        propose: Callable[[int | None], SglangPrefillProposal | AddReqResult],
        *,
        chunk_tokens_limit: int | None,
    ) -> SglangPrefillProposal | AddReqResult:
        """Prepare candidates together, preserving backing until the formed batch allocates."""

        reconciler = current_capacity_reconciler()

        def probe(limit: int | None) -> tuple[SglangPrefillProposal | AddReqResult, bool]:
            proposed = propose(limit)
            if proposed is AddReqResult.CONTINUE:
                raise RuntimeError("xpool elastic context state selection returned a post-commit verdict")
            outcome = torch.tensor(
                [0 if proposed is AddReqResult.NO_TOKEN else 1 if isinstance(proposed, AddReqResult) else 2],
                dtype=torch.int32,
            )
            if get_parallel().attn_tp_size > 1:
                torch.distributed.all_reduce(
                    outcome, op=torch.distributed.ReduceOp.MIN, group=reconciler.tp_group.cpu_group
                )
            result = int(outcome.item())
            if result < 2:
                return (AddReqResult.NO_TOKEN if result == 0 else AddReqResult.OTHER), False
            if isinstance(proposed, AddReqResult):
                raise RuntimeError("xpool elastic context state proposal disagrees with its common outcome")
            return proposed, reconciler.memory_ready(proposed.memory)

        original, ready = probe(chunk_tokens_limit)
        if isinstance(original, AddReqResult):
            if self.can_run_list or (
                original is AddReqResult.NO_TOKEN and not reconciler.vote_ready(reconciler.pending_retirement is None)
            ):
                return AddReqResult.OTHER
            return original
        selected = original if ready else None
        if not ready:
            lower, upper = 1, min(chunk_tokens_limit or 0, original.admission.extend_len - 1)
            while lower <= upper:
                middle = (lower + upper) // 2
                candidate, ready = probe(middle)
                if candidate is AddReqResult.OTHER:
                    lower = middle + 1
                elif ready and isinstance(candidate, SglangPrefillProposal):
                    selected = candidate
                    lower = middle + 1
                else:
                    upper = middle - 1
        if selected is not None and reconciler.prepare_prefill(selected):
            return selected
        if self.can_run_list:
            return AddReqResult.OTHER

        blocked = original if selected is None else selected
        advanced = reconciler.reclaim_requirement(blocked.memory, blocked.preparation_units)
        if selected is not None or advanced or not reconciler.vote_ready(reconciler.pending_retirement is None):
            return AddReqResult.OTHER
        return AddReqResult.NO_TOKEN

    def add_one_req(self, req: Req, has_chunked_req: bool, truncation_align_size: int | None) -> AddReqResult:
        """Agree before persistent mutation, then use the engine's ordinary commit."""

        reconciler = current_capacity_reconciler()
        attempt = SglangAllocation.current()
        if attempt is None:
            raise RuntimeError("xpool Prefill admission requires an allocation attempt")
        attempt.watch((req,))
        max_new = self._swa_new_tokens(req)
        remaining = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
        total = remaining + max_new + self.page_size
        candidate_tokens = remaining if self.rem_chunk_tokens is None else min(remaining, max(0, self.rem_chunk_tokens))
        selected_memory = self.proposal_for_admission(
            req, _PrefillAdmission(len(req.prefix_indices), candidate_tokens, max_new, candidate_tokens < remaining)
        ).memory

        def propose(limit: int | None) -> SglangPrefillProposal | AddReqResult:
            nonlocal selected_memory
            if attempt.prior_chunk_owner is not None or self.new_chunked_req is not None:
                return AddReqResult.OTHER
            if reconciler.draining:
                return AddReqResult.NO_TOKEN
            if self.prefill_max_requests is not None and len(self.can_run_list) >= self.prefill_max_requests:
                return AddReqResult.OTHER
            original_limit = self.rem_chunk_tokens
            self.rem_chunk_tokens = limit
            try:
                admission = super(ElasticPrefillAdder, self)._select_prefill_admission(
                    req,
                    total_tokens=total,
                    host_hit_length=0,
                    swa_host_hit_length=0,
                    truncation_align_size=truncation_align_size,
                )
                if isinstance(admission, AddReqResult):
                    return admission
                proposal = self.proposal_for_admission(req, admission)
                selected_memory = proposal.memory
                return proposal
            finally:
                self.rem_chunk_tokens = original_limit

        with self._lock_node(req.last_node):
            result = self.coordinate_prefill(propose, chunk_tokens_limit=self.rem_chunk_tokens)
            if isinstance(result, AddReqResult):
                reconciler.record_prefill_requirement(req, selected_memory if result is AddReqResult.NO_TOKEN else None)
                return result
            if self.prefill_delayer_single_pass is not None and not (
                self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                    local_prefillable=True,
                    running_batch=self.running_batch.batch_size(),
                    max_prefill_bs=self.max_prefill_bs,
                    max_running_requests=cast(int, self.max_running_requests),
                    waiting_queue_len=self.waiting_queue_len,
                )
            ):
                reconciler.record_prefill_requirement(req, None)
                return AddReqResult.OTHER
            reconciler.commit_prefill(req)
            self._commit_prefill_admission(req, result.admission, 0)
        reconciler.record_prefill_requirement(req, None)
        return self.budget_state()

    def add_chunked_req(self, req: Req) -> Req | None:
        """Keep continuation ownership on a miss; retain its distinct engine commit path."""

        reconciler = current_capacity_reconciler()
        attempt = SglangAllocation.current()
        if attempt is None:
            raise RuntimeError("xpool chunk continuation requires an allocation attempt")
        attempt.watch((req,))
        remaining = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
        selected_memory: SglangMemoryRequirement | None = None

        def propose(limit: int | None) -> SglangPrefillProposal | AddReqResult:
            nonlocal selected_memory
            budget = min(remaining if limit is None else limit, int(self.rem_total_tokens))
            if self.is_hybrid_swa:
                budget = min(budget, self.rem_swa_tokens - self.page_size)
            if budget <= 0:
                return AddReqResult.NO_TOKEN
            extent = min(remaining, budget)
            truncated = extent < remaining
            admission = _PrefillAdmission(
                len(req.prefix_indices),
                extent,
                # The scheduler normalizes the sampling limit before admission.
                0 if truncated else min(cast(int, req.sampling_params.max_new_tokens), CLIP_MAX_NEW_TOKENS),
                truncated,
            )
            proposal = self.proposal_for_admission(req, admission, continuation=True)
            selected_memory = proposal.memory
            return proposal

        result = self.coordinate_prefill(propose, chunk_tokens_limit=self.rem_chunk_tokens)
        if isinstance(result, AddReqResult):
            reconciler.record_prefill_requirement(req, selected_memory if result is AddReqResult.NO_TOKEN else None)
            return req
        if self.prefill_delayer_single_pass is not None:
            self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=cast(int, self.max_running_requests),
                waiting_queue_len=self.waiting_queue_len,
            )
        admission = result.admission
        reconciler.commit_prefill(req)
        req.set_extend_range(admission.prefix_len, admission.prefix_len + admission.extend_len)
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            admission.extend_len,
            admission.max_new_tokens,
            req.retracted_stain,
            is_chunked_continuation=True,
            compute_charge=admission.extend_len if self.exact_chunk_fill else None,
        )
        reconciler.record_prefill_requirement(req, None)
        return req if admission.is_chunked else None


def decode_requirement(batch: ScheduleBatch, selected_indices: list[int] | None = None) -> SglangMemoryRequirement:
    return current_capacity_reconciler().state.decode_requirement(batch, selected_indices)


def check_decode_mem(batch: ScheduleBatch, selected_indices: list[int] | None = None) -> bool:
    """Coordinate a full batch; repeated subset queries stay local and side-effect-free."""

    reconciler = current_capacity_reconciler()
    memory = decode_requirement(batch, selected_indices)
    if selected_indices is not None:
        forecast = reconciler.state.required_bytes(memory)
        immediate = reconciler.state.allocation_bytes(memory.allocation_units)
        return (
            forecast is not None
            and immediate is not None
            and max(forecast, immediate) <= reconciler.state.quota.limit_bytes
        )
    ready = reconciler.coordinate_decode(memory)
    reconciler.record_decode_requirement(batch.reqs, None if ready else memory)
    return ready


def release_retracted_request(batch: ScheduleBatch, index: int, remaining_count: int) -> bool:
    """Use original release semantics; beam groups abort instead of reentering Prefill."""

    request = batch.reqs[index]
    if request.beam_group is not None:
        request.to_finish = FINISH_ABORT(
            "Beam search group aborted: KV cache pool is full. Beam groups cannot be retracted.",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        free_member_rows(request.beam_group, batch.req_to_token_pool, batch.token_to_kv_pool_allocator)
        batch.release_req(index, remaining_count, offload_kv=False)
        return False
    if batch.release_req(index, remaining_count):
        return True
    request.to_finish = FINISH_ABORT(
        "Retraction host KV pool exhausted. Aborting the request.", status_code=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    return False


def retract_decode(batch: ScheduleBatch) -> tuple[list[Req], float, list[Req]]:
    """Retain victim order and coordinate each subset outside the pure checker."""

    reconciler = current_capacity_reconciler()
    order = beam_retraction_order(ScheduleBatch._get_decode_retraction_order(batch.reqs), batch.reqs)
    retracted: list[Req] = []
    aborted: list[Req] = []
    first = True
    while first or not reconciler.coordinate_decode(decode_requirement(batch, order)):
        if len(order) <= 1:
            break
        first = False
        index = order.pop()
        target = retracted if release_retracted_request(batch, index, len(order)) else aborted
        target.append(batch.reqs[index])
    if order and not reconciler.coordinate_decode(decode_requirement(batch, order)):
        index = order.pop()
        target = retracted if release_retracted_request(batch, index, 0) else aborted
        target.append(batch.reqs[index])
    batch.filter_batch(keep_indices=order)
    return retracted, NewTokenRatioTracker.estimate_new_token_ratio_after_retract(batch.reqs), aborted


def prepare_for_decode(batch: ScheduleBatch) -> None:
    """Stage the real dataclass and publish bookkeeping only after common allocation.

    The supported ordinary decoder path preserves overlap's borrowed sequence
    tensors. Sampling/penalty state is shared and is mutated only after READY.
    """

    reconciler = current_capacity_reconciler()
    scheduler = reconciler.scheduler
    if scheduler is None:
        raise RuntimeError("xpool Decode preparation requires the bound Scheduler")
    staged = reconciler.state.stage_decode(batch)
    strip_beam_tail(staged)
    append_beam_tail(staged)
    with reconciler.state.allocation() as attempt:
        attempt.watch(batch.reqs)
        try:
            if staged.model_config.is_encoder_decoder:
                staged.prepare_encoder_info_decode()
            staged.out_cache_loc = alloc_for_decode(staged, token_per_req=1)
        except SglangCapacityMiss:
            # The complete allocator has already restored unpublished owners.
            # Keep committed releases and return ordinary work to the same queue.
            order = beam_retraction_order(ScheduleBatch._get_decode_retraction_order(batch.reqs), batch.reqs)
            for remaining, index in reversed(tuple(enumerate(order))):
                request = batch.reqs[index]
                if release_retracted_request(batch, index, remaining):
                    scheduler._add_request_to_queue(request, is_retracted=True)
                else:
                    reason = cast(FINISH_ABORT, request.to_finish)
                    scheduler.ipc_channels.send_to_tokenizer.send_output(
                        _make_abort_req(request, finished_reason=reason.to_json()), request
                    )
                    scheduler.beam_coordinator.retire_group(request)
            batch.filter_batch(keep_indices=[])
            batch.batch_is_full = False
            return
        if not attempt.committed:
            raise RuntimeError("xpool Decode allocation returned without a common commit")
    penalizer = staged.sampling_info.penalizer_orchestrator
    if penalizer is not None and penalizer.is_required:
        staged.cumulate_penalty_output_tokens()
    for request in staged.reqs:
        request.decode_batch_idx += 1
    staged.seq_lens = staged.seq_lens + 1
    staged.seq_lens_cpu = staged.seq_lens_cpu + 1
    staged.orig_seq_lens = staged.orig_seq_lens + 1
    # The pinned engine uses None to invalidate this lazily computed int.
    staged.seq_lens_sum = None  # ty: ignore[invalid-assignment]
    reconciler.state.finish_decode(staged)
    for declaration in fields(staged):
        setattr(batch, declaration.name, getattr(staged, declaration.name))
