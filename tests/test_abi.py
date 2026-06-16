from __future__ import annotations

import subprocess
from pathlib import Path

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
        input_ptr=0x1000,
        output_ptr=0x2000,
        scratch_ptr=0x3000,
        num_tokens=4,
        hidden_size=2048,
        slot_id=5,
    )

    assert FfnRequestDescriptor.unpack(descriptor.pack()) == descriptor
    assert FfnRequestDescriptor.byte_size() == 80


def test_ffn_result_descriptor_round_trip() -> None:
    descriptor = FfnResultDescriptor(
        sequence=7,
        status=DescriptorStatus.DONE,
        error_code=0,
        slot_id=5,
        output_ptr=0x2000,
    )

    assert FfnResultDescriptor.unpack(descriptor.pack()) == descriptor
    assert FfnResultDescriptor.byte_size() == 40


def test_native_header_size_parity(tmp_path: Path) -> None:
    source = tmp_path / "abi_size.cc"
    binary = tmp_path / "abi_size"
    source.write_text(
        """
#include <iostream>
#include <abi.hpp>

int main() {
  std::cout << xpool::kFfnRequestDescriptorBytes << " "
            << xpool::kFfnResultDescriptorBytes << "\\n";
  return 0;
}
""".strip()
    )
    subprocess.run(
        ["c++", "-std=c++20", "-Isrc/cext/include", str(source), "-o", str(binary)],
        check=True,
        text=True,
    )

    output = subprocess.check_output([str(binary)], text=True).strip()
    assert output == f"{FfnRequestDescriptor.byte_size()} {FfnResultDescriptor.byte_size()}"
