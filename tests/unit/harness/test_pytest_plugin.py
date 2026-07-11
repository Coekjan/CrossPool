"""Behavior tests for xpool's pytest requirement plugin."""

import pytest

pytest_plugins = ["pytester"]


def test_weights_marker_requires_config(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_model_weights("model-a")
        def test_weights():
            pass
        """
    )

    result = pytester.runpytest("-p", "tests.harness.pytest_plugin", "--collect-only", "-q")

    result.stderr.fnmatch_lines(["*requires_model_weights must be paired with requires_config*"])


def test_missing_config_skips_by_default_and_fails_when_strict(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_config
        def test_config():
            pass
        """
    )

    default_result = pytester.runpytest("-p", "tests.harness.pytest_plugin", "-q")
    default_result.assert_outcomes(skipped=1)
    strict_result = pytester.runpytest(
        "-p",
        "tests.harness.pytest_plugin",
        "--strict-requirements",
        "-q",
    )
    strict_result.assert_outcomes(errors=1)


def test_deselected_requirement_is_not_resolved(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_config
        def test_config():
            pass

        def test_plain():
            pass
        """
    )

    result = pytester.runpytest("-p", "tests.harness.pytest_plugin", "-k", "plain", "-q")

    result.assert_outcomes(passed=1, deselected=1)
