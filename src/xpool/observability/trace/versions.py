"""Qualified offline trace toolchain; update processor release and digest together."""

from typing import Final

perfetto_version: Final = "0.58.2"
processor_version: Final = "v57.2"
processor_sha256: Final = "55ba613fc6d4f71df81eee2dbfc293020063655c241b3e314bff75345b802684"
processor_url: Final = (
    "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/"
    f"{processor_version}/linux-amd64/trace_processor_shell"
)
