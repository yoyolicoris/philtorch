"""The docstring examples that run Triton kernels, checked on CUDA machines only.

Apart from test_docstrings.py, so that the CUDA workflow, which runs on changes
to this module, doesn't run on changes to the others.
"""

import doctest
import importlib

import pytest
import torch

# By import_module, as philtorch.align.dtw is also the name of a function.
CUDA_MODULES = [
    importlib.import_module(name)
    for name in ("philtorch.align.ctc", "philtorch.align.dtw", "philtorch.estimation.hmm")
]


@pytest.mark.cuda
@pytest.mark.parametrize("module", CUDA_MODULES, ids=lambda module: module.__name__)
def test_docstring_examples(module):
    # Like PyTorch's, the examples assume torch is already imported.
    results = doctest.testmod(
        module, extraglobs={"torch": torch}, optionflags=doctest.NORMALIZE_WHITESPACE
    )
    assert results.attempted > 0, f"{module.__name__} has no docstring examples"
    assert results.failed == 0
