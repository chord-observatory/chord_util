"""
General CHORD utilities
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chord_util")
except PackageNotFoundError:
    # package is not installed
    pass
del version, PackageNotFoundError
