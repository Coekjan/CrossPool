"""xpool system."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xpool")
except PackageNotFoundError:
    __version__ = "0.0.0"
