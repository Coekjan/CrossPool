import json
import subprocess
from pathlib import Path
from typing import Literal

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from xbench.harness.serving.execution import capture_owned_metadata
from xbench.harness.serving.workload import PromptValue, validate_prompt
from xpool.integrations.sglang.devkit.requests import LocalModelMetadata, SglangRequestLimits
from xpool.model import ModelId


def test_offline_metadata_validates_actual_vocabulary_special_ids_and_text_limits(tmp_path: Path) -> None:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3, "outside": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "vocab_size": 4, "n_positions": 16}), encoding="utf-8"
    )
    metadata = LocalModelMetadata.from_checkpoint(tmp_path, text_prompts={"p": "a b"})
    assert metadata.admissible_ids == (2, 3)
    assert metadata.vocab_size == 4 and metadata.limits.context_length == 16
    assert metadata.text_lengths == {"p": 2}
    validate_prompt(PromptValue(prompt_id="p", input_ids=(1, 2)), metadata)
    with pytest.raises(ValueError, match="vocabulary"):
        validate_prompt(PromptValue(prompt_id="p", input_ids=(4,)), metadata)
    with pytest.raises(ValueError, match="input limit"):
        LocalModelMetadata.from_checkpoint(tmp_path, text_prompts={"p": " ".join(["a"] * 17)})
    with pytest.raises(ValueError, match="vocabulary"):
        LocalModelMetadata.from_checkpoint(tmp_path, text_prompts={"p": "outside"})


def test_static_service_limits_enforce_input_reserves_and_paged_group_ceiling() -> None:
    limits = SglangRequestLimits.from_server_info(
        json.dumps({"max_req_input_len": 18, "max_total_num_tokens": 24, "page_size": 4, "dcp_size": 1}).encode(),
        context_length=32,
    )
    limits.validate_request(9, 7)
    # The next output token reaches the group ceiling after page rounding.
    with pytest.raises(ValueError, match="static service limits"):
        limits.validate_request(9, 8)
    with pytest.raises(ValueError, match="static service limits"):
        limits.validate_request(18, 1)
    model_limits = SglangRequestLimits.from_context(32)
    model_limits.validate_request(25, 5)
    with pytest.raises(ValueError, match="static service limits"):
        model_limits.validate_request(25, 6)


def test_owned_hardware_capture_is_lease_scoped_and_failure_keeps_known_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def query(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "0, GPU-a, A100, 0000:01:00.0, 40960, 580.65\n"
                "1, GPU-b, H100, 0000:02:00.0, 81920, 580.65\n"
                "2, GPU-c, H100, 0000:03:00.0, 81920, 580.65\n"
                if command[1].startswith("--query-gpu")
                else "\x1b[4mGPU0 GPU1 GPU2 NIC0 CPU Affinity NUMA Affinity GPU NUMA ID\x1b[0m\n"
                "GPU0 X SYS SYS PIX 0-3 0 N/A\n"
                "GPU1 SYS X NV12 PIX 4-7 1 N/A\n"
                "GPU2 SYS NV12 X PIX 4-7 1 N/A\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", query)
    errors: list[str] = []
    placements: dict[ModelId, tuple[str, ...]] = {ModelId("test/one"): ("GPU-c",), ModelId("test/two"): ("GPU-c",)}
    roles: dict[Literal["atn", "ffn"], tuple[str, ...]] = {"atn": ("GPU-c",), "ffn": ("GPU-b",)}
    metadata = capture_owned_metadata(
        ("GPU-c", "GPU-b"), target_device_uuids=placements, role_device_uuids=roles, errors=errors
    )
    assert errors == [] and metadata.devices is not None and metadata.links is not None
    assert [device.uuid for device in metadata.devices] == ["GPU-c", "GPU-b"]
    assert metadata.devices[0].name == "H100" and metadata.devices[0].total_memory_bytes == 81920 * 1024 * 1024
    assert metadata.devices[0].pci_bus_id == "0000:03:00.0"
    assert (metadata.devices[0].cpu_affinity, metadata.devices[0].numa_affinity) == ("4-7", "1")
    assert {(link.source_uuid, link.destination_uuid, link.link) for link in metadata.links} == {
        ("GPU-b", "GPU-c", "NV12"),
        ("GPU-c", "GPU-b", "NV12"),
    }
    assert metadata.target_device_uuids == placements and metadata.role_device_uuids == roles
    assert metadata.driver_version == "580.65" and metadata.cuda_build_version is None

    def unavailable(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(subprocess, "run", unavailable)
    unknown = capture_owned_metadata(
        ("GPU-c", "GPU-b"), target_device_uuids=placements, role_device_uuids=roles, errors=errors
    )
    assert errors and unknown.devices is not None
    assert unknown.devices[0].uuid == "GPU-c" and unknown.devices[0].name is None
    assert unknown.links is None and unknown.driver_version is None
    assert unknown.target_device_uuids == placements
