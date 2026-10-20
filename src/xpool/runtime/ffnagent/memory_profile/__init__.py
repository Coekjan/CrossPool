"""Offline FFN memory profiling composition surface."""

from xpool.runtime.ffnagent.memory_profile.evidence import MemoryProfileEvidence
from xpool.runtime.ffnagent.memory_profile.runner import profile_ffn_memory

__all__ = ["MemoryProfileEvidence", "profile_ffn_memory"]
