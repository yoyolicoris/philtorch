from pathlib import Path
import torch

from . import _C

__version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text()
