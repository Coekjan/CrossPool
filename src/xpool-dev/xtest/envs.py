"""Internal environment declarations owned by native test execution."""

from xkit.envs import EnvironmentVariable

__all__ = ["CTEST_CANONICAL", "CTEST_INFRASTRUCTURE_SENTINEL"]

CTEST_CANONICAL = EnvironmentVariable("XTEST_CTEST_CANONICAL")
CTEST_INFRASTRUCTURE_SENTINEL = EnvironmentVariable("XTEST_CTEST_INFRASTRUCTURE_SENTINEL")
