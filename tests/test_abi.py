from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from xpool.abi import DescriptorStatus, FfnRequestDescriptor, FfnResultDescriptor, ForwardMode, TensorDType


def test_ffn_request_descriptor_round_trip() -> None:
    descriptor = FfnRequestDescriptor(
        sequence=7,
        instance_id=1,
        model_id=2,
        layer_id=3,
        forward_mode=ForwardMode.DECODE,
        dtype=TensorDType.BF16,
        status=DescriptorStatus.PUBLISHED,
        input_offset=0x1000,
        output_offset=0x2000,
        scratch_offset=0x3000,
        num_tokens=4,
        hidden_size=2048,
        slot_id=5,
    )

    assert FfnRequestDescriptor.unpack(descriptor.pack()) == descriptor
    assert FfnRequestDescriptor.byte_size() == 80


def test_ffn_request_descriptor_rejects_wrong_payload_length() -> None:
    descriptor = FfnRequestDescriptor(
        sequence=7,
        instance_id=1,
        model_id=2,
        layer_id=3,
        forward_mode=ForwardMode.DECODE,
        dtype=TensorDType.BF16,
        status=DescriptorStatus.PUBLISHED,
        input_offset=0x1000,
        output_offset=0x2000,
        scratch_offset=0x3000,
        num_tokens=4,
        hidden_size=2048,
    )

    try:
        FfnRequestDescriptor.unpack(descriptor.pack()[:-1])
    except ValueError as exc:
        assert "payload length" in str(exc)
    else:
        raise AssertionError("expected truncated request descriptor to fail")


def test_ffn_request_descriptor_rejects_reserved_bits() -> None:
    descriptor = FfnRequestDescriptor(
        sequence=7,
        instance_id=1,
        model_id=2,
        layer_id=3,
        forward_mode=ForwardMode.DECODE,
        dtype=TensorDType.BF16,
        status=DescriptorStatus.PUBLISHED,
        input_offset=0x1000,
        output_offset=0x2000,
        scratch_offset=0x3000,
        num_tokens=4,
        hidden_size=2048,
        reserved=1,
    )

    try:
        FfnRequestDescriptor.unpack(descriptor.pack())
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("expected reserved request descriptor field to fail")


def test_ffn_result_descriptor_round_trip() -> None:
    descriptor = FfnResultDescriptor(
        sequence=7,
        status=DescriptorStatus.DONE,
        error_code=0,
        slot_id=5,
        output_offset=0x2000,
    )

    assert FfnResultDescriptor.unpack(descriptor.pack()) == descriptor
    assert FfnResultDescriptor.byte_size() == 40


def test_ffn_result_descriptor_rejects_wrong_payload_length() -> None:
    descriptor = FfnResultDescriptor(
        sequence=7,
        status=DescriptorStatus.DONE,
        error_code=0,
        slot_id=5,
        output_offset=0x2000,
    )

    try:
        FfnResultDescriptor.unpack(descriptor.pack()[:-1])
    except ValueError as exc:
        assert "payload length" in str(exc)
    else:
        raise AssertionError("expected truncated result descriptor to fail")


def test_ffn_result_descriptor_rejects_reserved_bits() -> None:
    descriptor = FfnResultDescriptor(
        sequence=7,
        status=DescriptorStatus.DONE,
        error_code=0,
        slot_id=5,
        reserved=1,
        output_offset=0x2000,
    )

    try:
        FfnResultDescriptor.unpack(descriptor.pack())
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("expected reserved result descriptor field to fail")


def test_forward_mode_values_match_native_contract() -> None:
    """ForwardMode integers are the on-wire ABI values fed to the native op.

    The SGLang shim forwards these ints straight to ``torch.ops.xpool.ffn_shim``,
    so they must equal the C++ ``xpool::ForwardMode`` enum in ``abi.hpp`` and
    never drift to a separate source of truth.
    """

    assert int(ForwardMode.DECODE) == 1
    assert int(ForwardMode.EXTEND) == 2


def test_native_header_size_parity(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("c++ compiler is not available")

    source = tmp_path / "abi_size.cc"
    binary = tmp_path / "abi_size"
    source.write_text(
        """
#include <iostream>
#include <abi.hpp>

int main() {
  std::cout << xpool::kFfnRequestDescriptorBytes << " "
            << xpool::kFfnResultDescriptorBytes << " "
            << static_cast<unsigned>(xpool::ForwardMode::kDecode) << " "
            << static_cast<unsigned>(xpool::ForwardMode::kExtend) << "\\n";
  return 0;
}
""".strip()
    )
    subprocess.run(
        [compiler, "-std=c++20", "-Isrc/cext/include", str(source), "-o", str(binary)],
        check=True,
        text=True,
    )

    output = subprocess.check_output([str(binary)], text=True).strip()
    request_bytes, result_bytes, decode, extend = output.split()
    assert int(request_bytes) == FfnRequestDescriptor.byte_size()
    assert int(result_bytes) == FfnResultDescriptor.byte_size()
    assert int(decode) == int(ForwardMode.DECODE)
    assert int(extend) == int(ForwardMode.EXTEND)
