"""How each prototype Helion kernel runs on the current GPU.

The kernels are tuned ahead of time (AOT) by ``scripts/helion_aot.py``, which
writes, for the GPU it runs on:

* ``_helion_aot_<module>_cuda_<sm>.py`` next to the kernels' module: a
  decision tree per kernel that picks a tuned configuration from the input
  shapes, which Helion reads at run time;
* ``_aot/<sm>/<kernel><variant>.py``: standalone Triton code for those
  configurations with the same decision tree, which needs no Helion.

For each GPU, :class:`Dispatch` runs a kernel from the first of these that
exists:

1. the standalone file for the GPU's compute capability;
2. the Helion kernel with a heuristic file for it, or for an older compute
   capability of the same vendor, which Helion falls back to by itself;
3. the Helion kernel with a fallback configuration chosen by hand.

Helion's own fallback, its default configuration, can take minutes to
compile for some kernels, so the last step uses ours instead.
"""

import importlib.util
import inspect
from pathlib import Path

import helion
import helion.language as hl
import torch

_STANDALONE_DIR = Path(__file__).parent / "_aot"


def compute_capability(device: torch.device) -> str:
    """The compute capability as Helion names it, e.g. "sm120"."""
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def variant_name(kernel: helion.Kernel, args: tuple) -> str:
    """A suffix naming the dtype and constexpr values, e.g. "__float32__soft1__diag0".

    Helion specializes a kernel's generated code on these, so every
    combination needs its own standalone file. The dtype is the first
    argument's: the kernels take a single floating dtype.
    """
    params = inspect.signature(kernel.fn).parameters.values()
    parts = [str(args[0].dtype).removeprefix("torch.")]
    for param, arg in zip(params, args):
        if param.annotation is hl.constexpr or param.annotation == "hl.constexpr":
            value = arg.value if isinstance(arg, hl.constexpr) else arg
            parts.append(f"{param.name}{int(value)}")
    return "".join(f"__{part}" for part in parts)


def standalone_path(kernel: helion.Kernel, args: tuple, cc: str) -> Path:
    return _STANDALONE_DIR / cc / f"{kernel.name}{variant_name(kernel, args)}.py"


def _load_standalone(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"philtorch_aot_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def _prime(kernel: helion.Kernel, args: tuple) -> None:
    """Bind ``kernel`` first to new tensors like ``args``, every size at least 2.

    Helion traces a kernel with its first call's arguments but keys the
    compiled code more loosely. With dynamic shapes, sizes 0 and 1 share the
    bucket of larger sizes, yet code traced with one row computes only part
    of a longer input; and arguments that alias each other compile to code
    that reads one of them for both. Code traced from distinct tensors with
    sizes of at least 2 is correct for every call, so the first bind uses
    those.
    """
    kernel.bind(
        tuple(
            arg.new_empty([max(size, 2) for size in arg.shape])
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in args
        )
    )


def _has_heuristic(kernel: helion.Kernel) -> bool:
    from helion.autotuner.aot_cache import find_heuristic_file

    return find_heuristic_file(kernel.fn.__code__.co_filename, kernel_name=kernel.name) is not None


class Dispatch:
    """Call ``kernel`` (an AOT Helion kernel) as described in the module docstring."""

    def __init__(self, kernel: helion.Kernel, fallback: helion.Config):
        self.kernel = kernel
        self.fallback = fallback
        self._fallback_kernel = None
        self._chosen: dict = {}  # (compute capability, variant) -> callable

    @property
    def name(self) -> str:
        return self.kernel.name

    def _choose(self, args: tuple, cc: str):
        path = standalone_path(self.kernel, args, cc)
        if path.exists():
            return _load_standalone(path, self.kernel.name)
        if _has_heuristic(self.kernel):
            kernel = self.kernel
        else:
            if self._fallback_kernel is None:
                self._fallback_kernel = helion.kernel(
                    self.kernel.fn, config=self.fallback, static_shapes=False
                )
            kernel = self._fallback_kernel
        _prime(kernel, args)
        return kernel

    def __call__(self, *args):
        cc = compute_capability(args[0].device)
        key = (cc, variant_name(self.kernel, args))
        fn = self._chosen.get(key)
        if fn is None:
            fn = self._chosen[key] = self._choose(args, cc)
        return fn(*args)
