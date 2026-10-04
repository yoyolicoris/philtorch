import doctest

import pytest
import torch

import philtorch.mat
import philtorch.poly
import philtorch.utils

# Modules whose docstring examples are checked. Add a module here once its
# docstrings have examples.
MODULES = [philtorch.mat, philtorch.poly, philtorch.utils]


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_docstring_examples(module):
    # Like PyTorch's, the examples assume torch is already imported.
    results = doctest.testmod(
        module, extraglobs={"torch": torch}, optionflags=doctest.NORMALIZE_WHITESPACE
    )
    assert results.attempted > 0, f"{module.__name__} has no docstring examples"
    assert results.failed == 0
