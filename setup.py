from setuptools import setup
import os
import glob
import subprocess
import sys

library_name = "philtorch"


def get_homebrew_prefix(formula):
    try:
        result = subprocess.run(
            ["brew", "--prefix", formula],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "Homebrew is required for macOS OpenMP builds; "
            "install Homebrew, then run 'brew install llvm libomp'."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Homebrew formula '{formula}' is required for macOS OpenMP builds; "
            "install it with 'brew install llvm libomp'."
        ) from error

    return result.stdout.strip()


def get_macos_openmp_config(extra_compile_args, extra_link_args, torch_lib):
    llvm_prefix = get_homebrew_prefix("llvm")
    libomp_prefix = get_homebrew_prefix("libomp")
    compiler = os.path.join(llvm_prefix, "bin", "clang++")
    include_dir = os.path.join(libomp_prefix, "include")

    required_paths = (
        compiler,
        os.path.join(include_dir, "omp.h"),
        os.path.join(torch_lib, "libomp.dylib"),
    )
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise RuntimeError(
            "macOS OpenMP build dependencies are incomplete; missing: "
            + ", ".join(missing_paths)
        )

    configured_compile_args = {
        **extra_compile_args,
        "cxx": [*extra_compile_args.get("cxx", ()), f"-I{include_dir}"],
    }
    configured_link_args = [
        *extra_link_args,
        f"-L{torch_lib}",
        f"-Wl,-rpath,{torch_lib}",
    ]
    return compiler, configured_compile_args, configured_link_args


def get_extensions():
    import torch
    from torch.utils.cpp_extension import (
        CppExtension,
        CUDAExtension,
        CUDA_HOME,
    )

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    use_openmp = (
        torch.backends.openmp.is_available()
        and os.environ.get("PHILTORCH_DISABLE_OPENMP", "0") != "1"
    )
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {}
    if use_openmp:
        if sys.platform == "win32":
            # host_dot.h uses `#pragma omp simd`, which requires MSVC's
            # experimental OpenMP mode; the legacy `/openmp` flag only
            # supports OpenMP 2.0 constructs (no `simd`) and fails to compile.
            extra_compile_args["cxx"] = ["/openmp:experimental"]
            # MSVC links OpenMP automatically — no extra_link_args entry needed
        else:
            extra_compile_args["cxx"] = ["-fopenmp"]
            extra_link_args.append("-fopenmp")
        if sys.platform == "darwin":
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            compiler, extra_compile_args, extra_link_args = get_macos_openmp_config(
                extra_compile_args, extra_link_args, torch_lib
            )
            os.environ.setdefault("CXX", compiler)

    this_dir = os.path.abspath(os.path.dirname(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    if sys.platform == "darwin":
        sources += list(glob.glob(os.path.join(extensions_dir, "*.mm")))
        extra_link_args.extend(["-framework", "Foundation", "-framework", "Metal"])

    if use_cuda:
        sources += cuda_sources
        extra_compile_args["nvcc"] = ["--extended-lambda"]

    if len(sources) == 0:
        return []

    ext_modules = [
        extension(
            f"{library_name}._C",
            # sources,
            [os.path.relpath(s, this_dir) for s in sources],
            # ["philtorch/csrc/recur2.cu"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


try:
    ext_modules = get_extensions()
except ImportError:
    ext_modules = []

if not ext_modules:
    setup()
else:
    from torch.utils.cpp_extension import BuildExtension

    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
