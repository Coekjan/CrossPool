from __future__ import annotations

import pytest

from tests.harness.support.kv import kv_capacity_profile


def test_kv_capacity_profile_projects_bundle_prefix_to_aligned_tokens() -> None:
    profile = kv_capacity_profile()

    assert [profile.usable_tokens(bundle_count) for bundle_count in (1, 2, 8)] == [0, 4, 28]
    with pytest.raises(ValueError, match="outside its reservation"):
        profile.usable_tokens(9)
