import glob
import os
import sys
import sysconfig

from setuptools import setup

# build_support.py sits next to setup.py; the setuptools.build_meta:__legacy__
# backend puts this directory on sys.path, which this import relies on.
from build_support import MIN_TORCH, resolve_cuda_build

library_name = "philtorch"
# The extension uses only CPython's limited API, so one abi3 wheel, tagged with
# the Python that builds it, serves that Python and every later one. abi3
# doesn't cover free-threaded CPython, which gets a regular wheel.
py_limited_api = not sysconfig.get_config_var("Py_GIL_DISABLED")


def write_metal_header(source, header):
    # torch's Metal shim compiles only from source text, so the extension
    # carries each shader as a raw string, whose delimiter it can't contain,
    # named after its file: mps_common.metal is MPS_COMMON_METAL.
    with open(source, encoding="utf-8") as file:
        text = file.read()
    name = os.path.basename(source).replace(".", "_").upper()
    os.makedirs(os.path.dirname(header), exist_ok=True)
    with open(header, "w", encoding="utf-8") as file:
        file.write(
            "#pragma once\n"
            f'constexpr const char *{name} = R"PHILTORCH_METAL({text})PHILTORCH_METAL";\n'
        )


def get_extensions():
    import torch
    from torch.utils.cpp_extension import (
        CUDA_HOME,
        CppExtension,
        CUDAExtension,
    )

    use_cuda = resolve_cuda_build(
        force_cuda_value=os.environ.get("PHILTORCH_FORCE_CUDA", "0"),
        cuda_home=CUDA_HOME,
        cuda_available=torch.cuda.is_available(),
        cuda_arch_list=os.environ.get("TORCH_CUDA_ARCH_LIST"),
        torch_cuda_version=torch.version.cuda,
    )
    print(
        "[philtorch build] "
        f"use_cuda={use_cuda} "
        f"CUDA_HOME={CUDA_HOME or 'None'} "
        f"torch={torch.__version__} "
        f"torch_cuda={torch.version.cuda or 'None'} "
        "TORCH_CUDA_ARCH_LIST="
        f"{os.environ.get('TORCH_CUDA_ARCH_LIST') or 'unset'}"
    )
    extension = CUDAExtension if use_cuda else CppExtension

    extra_compile_args = {}
    # The CPU kernels run on torch's threads (torch::stable::parallel_for), so
    # the extension needs no OpenMP runtime: only host_dot.h's
    # `#pragma omp simd`, which -fopenmp-simd enables. MSVC has no such flag;
    # its experimental OpenMP mode takes `simd`, which the legacy `/openmp`
    # (OpenMP 2.0) rejects.
    if os.environ.get("PHILTORCH_DISABLE_OPENMP", "0") != "1":
        extra_compile_args["cxx"] = (
            ["/openmp:experimental"] if sys.platform == "win32" else ["-fopenmp-simd"]
        )

    this_dir = os.path.abspath(os.path.dirname(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))
    if use_cuda and not cuda_sources:
        raise RuntimeError(
            f"CUDA compilation was selected, but no .cu sources were found in {extensions_dir}."
        )

    # torch.__version__ is a TorchVersion, which compares as a version.
    if torch.__version__ < MIN_TORCH:
        raise RuntimeError(
            f"philtorch's ops target the stable ABI of PyTorch {MIN_TORCH}, so they "
            f"build with PyTorch {MIN_TORCH} or newer; found {torch.__version__}."
        )

    include_dirs = []
    if sys.platform == "darwin":
        # .mm marks the macOS-only sources, which reach Metal through torch's
        # C shim and include their shaders, written below as headers.
        sources += list(glob.glob(os.path.join(extensions_dir, "*.mm")))
        metal_dir = os.path.join(this_dir, "build", "metal")
        include_dirs.append(metal_dir)
        for shader in glob.glob(os.path.join(extensions_dir, "*.metal")):
            name = os.path.basename(shader)
            write_metal_header(shader, os.path.join(metal_dir, f"{name}.h"))

    if use_cuda:
        sources += cuda_sources
        extra_compile_args.setdefault("nvcc", []).append("--extended-lambda")
        # Store the GPU code (one copy per target architecture) compressed in
        # the binary, as PyTorch does for its own wheels. It is otherwise most
        # of the extension's size, and is decompressed once at load.
        extra_compile_args.setdefault("nvcc", []).extend(["-Xfatbin", "-compress-all"])

    if len(sources) == 0:
        return []

    ext_modules = [
        extension(
            f"{library_name}._C",
            [os.path.relpath(s, this_dir) for s in sources],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


try:
    ext_modules = get_extensions()
except ImportError:
    # Only torch's absence is treated as "maybe metadata-only"; other errors
    # from get_extensions() (e.g. a torch older than MIN_TORCH) propagate as-is,
    # since those are real build failures rather than a missing-torch case.
    #
    # Metadata-only invocations (e.g. `setup.py egg_info`/`sdist`, which pip
    # also runs to prepare metadata/sdists in an isolated build environment
    # populated solely from build-system.requires) don't need torch to be
    # importable. Let those keep working without torch pre-installed.
    #
    # Actually building an extension (bdist_wheel/build_ext/develop/install)
    # does need torch; failing loudly here instead of silently degrading to
    # an extension-less wheel avoids shipping a philtorch that's missing
    # `_C` with no indication anything went wrong.
    metadata_only_commands = {"egg_info", "sdist", "dist_info"}
    if metadata_only_commands.intersection(sys.argv):
        ext_modules = []
    else:
        raise RuntimeError(
            "philtorch could not `import torch` while building its "
            "C++/CUDA extension. Install torch first (see README), then "
            "reinstall/rebuild with `--no-build-isolation` so the build "
            "sees it."
        )

# See MIN_TORCH in build_support.py.
install_requires = [f"torch >= {MIN_TORCH}.0"]

if not ext_modules:
    setup(install_requires=install_requires)
else:
    from torch.utils.cpp_extension import BuildExtension

    setup(
        install_requires=install_requires,
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": f"cp3{sys.version_info.minor}"}}
        if py_limited_api
        else {},
    )
