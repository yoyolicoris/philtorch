import doctest
import importlib

import pytest
import torch

import philtorch.estimation.kalman
import philtorch.lpv.filtering
import philtorch.lpv.recur
import philtorch.lpv.ssm
import philtorch.lti.delay
import philtorch.lti.filtering
import philtorch.lti.interp
import philtorch.lti.recur
import philtorch.lti.ssm
import philtorch.mat
import philtorch.poly
import philtorch.utils

# Modules whose docstring examples are checked. Add a module here once its
# docstrings have examples.
MODULES = [
    philtorch.mat,
    philtorch.poly,
    philtorch.utils,
    philtorch.lti.delay,
    philtorch.lti.filtering,
    philtorch.lti.interp,
    philtorch.lti.recur,
    philtorch.lti.ssm,
    philtorch.lpv.filtering,
    philtorch.lpv.recur,
    philtorch.lpv.ssm,
    philtorch.estimation.kalman,
]

# Modules whose examples run Triton kernels, checked on CUDA machines only.
# By import_module, as philtorch.align.dtw is also the name of a function.
CUDA_MODULES = [
    importlib.import_module(name) for name in ("philtorch.align.ctc", "philtorch.align.dtw")
]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.mark.parametrize(
    "module",
    MODULES + [pytest.param(module, marks=requires_cuda) for module in CUDA_MODULES],
    ids=lambda module: module.__name__,
)
def test_docstring_examples(module):
    # Like PyTorch's, the examples assume torch is already imported.
    results = doctest.testmod(
        module, extraglobs={"torch": torch}, optionflags=doctest.NORMALIZE_WHITESPACE
    )
    assert results.attempted > 0, f"{module.__name__} has no docstring examples"
    assert results.failed == 0
