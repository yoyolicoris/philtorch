from pathlib import Path
import warnings
import torch

try:
    from . import _C

    EXTENSION_LOADED = True
except ImportError:
    EXTENSION_LOADED = False
    warnings.warn("Custom extension not loaded.")

__version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text()
