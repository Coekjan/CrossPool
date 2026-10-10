"""Offline model facts and pinned SGLang request-budget constraints."""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from sglang.srt.utils.hf_transformers_utils import get_context_length, get_hf_text_config, get_tokenizer
from transformers import AutoConfig

__all__ = ["LocalModelMetadata", "SglangRequestLimits"]


class SglangRequestLimits(BaseModel):
    """Static acceptance bounds, independent of the active Elastic KV prefix.

    The scheduler requires input below max_req_input_len, reserves one token
    below max_req_len, and requires one free page beyond the paged request.
    Offline bounds include the model-derived reserves; server bounds also use
    the startup Capacity Group ceiling returned by /server_info.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    context_length: int = Field(gt=0)
    max_req_input_len: int = Field(gt=0)
    max_total_num_tokens: int | None = Field(default=None, gt=0)
    page_size: int = Field(default=1, gt=0)
    dcp_size: int = Field(default=1, gt=0)

    @classmethod
    def from_context(cls, context_length: int) -> Self:
        """Apply the pinned worker's model-only request and input reserves."""
        return cls(context_length=context_length, max_req_input_len=context_length - 6)

    @classmethod
    def from_server_info(cls, contents: bytes, *, context_length: int) -> Self:
        """Read the first scheduler's startup limits, not a DP aggregate."""
        info = TypeAdapter(dict[str, JsonValue]).validate_json(contents)
        return cls.model_validate(
            {
                "context_length": context_length,
                "max_req_input_len": info["max_req_input_len"],
                "max_total_num_tokens": info["max_total_num_tokens"],
                "page_size": info["page_size"],
                "dcp_size": info["dcp_size"],
            }
        )

    def input_budget(self, output_tokens: int) -> int:
        """Return the model/request-bound maximum input for an output budget."""
        return min(
            self.max_req_input_len - 1, self.context_length - output_tokens, self.max_req_input_len + 4 - output_tokens
        )

    def output_budget(self, input_tokens: int) -> int:
        """Return the maximum output preserving request and paged-admission bounds."""
        budget = min(self.context_length - input_tokens, self.max_req_input_len + 4 - input_tokens)
        if self.max_total_num_tokens is not None:
            paged_input = -(-input_tokens // self.page_size) * self.page_size
            budget = min(budget, self.max_total_num_tokens * self.dcp_size - paged_input - self.page_size - 1)
        return budget

    def validate_request(self, input_tokens: int, output_tokens: int) -> None:
        """Reject a prepared request that the engine would reject or shorten."""
        if input_tokens > self.input_budget(output_tokens) or output_tokens > self.output_budget(input_tokens):
            raise ValueError(
                f"input={input_tokens}, output={output_tokens} exceeds static service limits "
                f"(context={self.context_length}, input_bound={self.max_req_input_len}, "
                f"kv_ceiling={self.max_total_num_tokens}, page_size={self.page_size}, dcp_size={self.dcp_size})"
            )


class LocalModelMetadata(BaseModel):
    """Local serving-tokenizer facts resolved once during workload preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    vocab_size: int = Field(gt=0)
    limits: SglangRequestLimits
    admissible_ids: tuple[Annotated[int, Field(ge=0)], ...]
    text_lengths: dict[str, Annotated[int, Field(ge=0)]]

    @classmethod
    def from_checkpoint(cls, path: Path, *, text_prompts: Mapping[str, str]) -> Self:
        """Load local files only, using SGLang's context and tokenizer semantics."""
        config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        tokenizer = get_tokenizer(str(path), local_files_only=True, trust_remote_code=True)
        text_config = get_hf_text_config(config)
        vocab_size = TypeAdapter(Annotated[int, Field(gt=0, strict=True)]).validate_python(text_config.vocab_size)
        context = get_context_length(text_config)
        special = set(tokenizer.all_special_ids)
        ids = tuple(sorted({id for id in tokenizer.get_vocab().values() if 0 <= id < vocab_size and id not in special}))
        lengths = {}
        for prompt_id, text in text_prompts.items():
            tokens = tokenizer.encode(text)
            if len(tokens) >= context:
                raise ValueError(f"prompt {prompt_id!r} exceeds the known model input limit")
            if any(token < 0 or token >= vocab_size for token in tokens):
                raise ValueError(f"prompt {prompt_id!r} contains IDs outside the local model vocabulary")
            lengths[prompt_id] = len(tokens)
        return cls(
            vocab_size=vocab_size,
            limits=SglangRequestLimits.from_context(context),
            admissible_ids=ids,
            text_lengths=lengths,
        )
