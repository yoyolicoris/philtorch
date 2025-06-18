from pathlib import Path

__version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text()
