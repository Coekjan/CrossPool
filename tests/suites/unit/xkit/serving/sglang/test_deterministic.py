from __future__ import annotations

import pytest
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.plugins.hook_registry import HookRegistry

import xkit.serving.sglang.deterministic


@pytest.mark.parametrize("fail", [False, True])
def test_decode_policy_restores_backend_after_return_or_failure(fail: bool) -> None:
    backend = object.__new__(FlashAttentionBackend)
    backend.fa_impl_ver = 3
    backend.use_mla = True
    backend.num_splits = 4

    def execute(current: FlashAttentionBackend) -> int:
        assert current.num_splits == 1
        if fail:
            raise RuntimeError("decode failed")
        return 17

    if fail:
        with pytest.raises(RuntimeError, match="decode failed"):
            xkit.serving.sglang.deterministic.around_decode(execute, backend)
    else:
        assert xkit.serving.sglang.deterministic.around_decode(execute, backend) == 17
    assert backend.num_splits == 4


@pytest.mark.parametrize(("version", "mla"), [(3, False), (4, True)])
def test_other_attention_paths_retain_their_policy(version: int, mla: bool) -> None:
    backend = object.__new__(FlashAttentionBackend)
    backend.fa_impl_ver = version
    backend.use_mla = mla
    backend.num_splits = 4

    def execute(current: FlashAttentionBackend) -> int:
        return current.num_splits

    assert xkit.serving.sglang.deterministic.around_decode(execute, backend) == 4


def test_automatic_plugin_discovery_does_not_enable_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)

    def register(*args: object, **kwargs: object) -> None:
        pytest.fail("automatic discovery installed the deterministic policy")

    monkeypatch.setattr(HookRegistry, "register", register)
    xkit.serving.sglang.deterministic.install()


def test_explicit_policy_rejects_an_unapplied_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_PLUGINS", "xkit_deterministic")
    monkeypatch.setattr(HookRegistry, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(HookRegistry, "apply_hooks", classmethod(lambda registry: None))
    monkeypatch.setattr(HookRegistry, "_patched", set())

    xkit.serving.sglang.deterministic.install()

    with pytest.raises(SystemExit, match="deterministic hook was not applied"):
        HookRegistry.apply_hooks()
