"""The ``cuda`` mark: tests that need a CUDA device.

Marked tests are skipped where no CUDA device is available, as on the CPU CI,
and the GPU CI runs exactly them, with ``-m cuda``.
"""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda: needs a CUDA device; skipped without one, run by the GPU CI"
    )
    # Autograd runs CUDA backward passes on a thread of its own, started in
    # C++, which coverage can't trace: the backwards would read as never run.
    # Run them on the calling thread, as on CPU.
    torch.autograd.set_multithreading_enabled(False)


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if item.get_closest_marker("cuda"):
            item.add_marker(skip)
