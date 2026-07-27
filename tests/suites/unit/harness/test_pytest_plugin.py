"""Behavior tests for xpool's pytest requirement plugin."""

import pytest

import tests.harness.test_plan

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


def test_mps_marker_rejects_arguments(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_mps(True)
        def test_mps():
            pass
        """
    )

    result = pytester.runpytest("-p", "tests.harness.pytest_plugin", "--collect-only", "-q")

    result.stderr.fnmatch_lines(["*requires_mps accepts no arguments*"])


def test_collection_requires_complete_token_parity_group(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "e2e"
    test_directory.mkdir(parents=True)
    (test_directory / "test_e2e_example.py").write_text(
        """
import pytest

@pytest.mark.token_parity_group(name="example", expected_case_count=2)
@pytest.mark.parametrize("mode", ["eager", "full"])
def test_e2e_example(mode):
    pass
""",
        encoding="utf-8",
    )

    result = pytester.runpytest(
        "-p",
        "tests.harness.pytest_plugin",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        "-o",
        "timeout=10",
        "-k",
        "eager",
        f"--xpool-test-plan={pytester.path / 'plan.json'}",
        str(test_directory),
    )

    assert result.ret != 0
    assert "token parity groups must contain every expected case" in result.stderr.str()


def test_collection_rejects_xfail_token_parity_case(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail(reason="not supported")
        @pytest.mark.token_parity_group(name="example", expected_case_count=2)
        def test_e2e_example():
            pass
        """
    )

    result = pytester.runpytest("-p", "tests.harness.pytest_plugin", "--collect-only", "-q")

    result.stderr.fnmatch_lines(["*token parity cases cannot use xfail*"])


def test_collection_worker_writes_final_typed_item_metadata(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "integration"
    test_directory.mkdir(parents=True)
    (test_directory / "test_example.py").write_text(
        """
import pytest

@pytest.mark.requires_cuda(min_devices=2)
@pytest.mark.requires_config
@pytest.mark.requires_mps
@pytest.mark.requires_model_weights("organization/model")
@pytest.mark.estimated_duration(seconds=3)
@pytest.mark.timeout(12)
def test_example():
    pass
""",
        encoding="utf-8",
    )
    output = pytester.path / "plan.json"

    result = pytester.runpytest(
        "-p",
        "tests.harness.pytest_plugin",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        f"--xpool-test-plan={output}",
        str(test_directory),
    )

    result.assert_outcomes()
    plan = tests.harness.test_plan.TestPlan.read(output)
    assert len(plan.cases) == 1
    case = plan.cases[0]
    assert case.path == "tests/suites/integration/test_example.py"
    assert case.stage is tests.harness.test_plan.TestStage.INTEGRATION
    assert case.requirements.cuda_count == 2
    assert case.requirements.requires_mps
    assert case.requirements.requires_config
    assert case.requirements.model_ids == ("organization/model",)
    assert case.estimated_duration_seconds == 3
    assert case.timeout_seconds == 12


def test_collection_worker_rejects_unbounded_item(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "unit"
    test_directory.mkdir(parents=True)
    (test_directory / "test_example.py").write_text("def test_example(): pass\n", encoding="utf-8")

    result = pytester.runpytest(
        "-p",
        "tests.harness.pytest_plugin",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        f"--xpool-test-plan={pytester.path / 'plan.json'}",
        str(test_directory),
    )

    result.stderr.fnmatch_lines(["*requires a timeout marker or configured pytest timeout*"])


def test_collection_worker_uses_configured_pytest_timeout(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    test_directory = pytester.path / "tests" / "suites" / "unit"
    test_directory.mkdir(parents=True)
    (test_directory / "test_example.py").write_text("def test_example(): pass\n", encoding="utf-8")
    output = pytester.path / "plan.json"

    result = pytester.runpytest(
        "-p",
        "tests.harness.pytest_plugin",
        f"--rootdir={pytester.path}",
        "--collect-only",
        "-q",
        "-o",
        "timeout=45",
        f"--xpool-test-plan={output}",
        str(test_directory),
    )

    result.assert_outcomes()
    (case,) = tests.harness.test_plan.TestPlan.read(output).cases
    assert case.timeout_seconds == 45
