"""The ``cuda`` mark: tests that need a CUDA device.

Marked tests are skipped where no CUDA device is available, as on the CPU CI,
and the GPU CI runs exactly them, with ``-m cuda``. The ``eight_threads``
fixture runs a test on 8 CPU threads, enough for the CPU recurrence kernel to
split one sequence over time.
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


@pytest.fixture
def eight_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(8)
    yield
    torch.set_num_threads(threads)
