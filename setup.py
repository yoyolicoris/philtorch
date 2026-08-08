from setuptools import setup
import os
import glob
import subprocess
import sys
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

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


def configure_macos_openmp(extra_compile_args, extra_link_args):
    llvm_prefix = get_homebrew_prefix("llvm")
    libomp_prefix = get_homebrew_prefix("libomp")
    compiler = os.path.join(llvm_prefix, "bin", "clang++")
    include_dir = os.path.join(libomp_prefix, "include")
    library_dir = os.path.join(libomp_prefix, "lib")

    required_paths = (
        compiler,
        os.path.join(include_dir, "omp.h"),
        os.path.join(library_dir, "libomp.dylib"),
    )
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise RuntimeError(
            "Homebrew LLVM/OpenMP installation is incomplete; missing: "
            + ", ".join(missing_paths)
        )

    os.environ.setdefault("CXX", compiler)
    extra_compile_args["cxx"].append(f"-I{include_dir}")
    extra_link_args.extend(
        [
            f"-L{library_dir}",
            f"-Wl,-rpath,{library_dir}",
        ]
    )


def get_extensions():
    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    use_openmp = torch.backends.openmp.is_available()
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {}
    if use_openmp:
        extra_compile_args["cxx"] = ["-fopenmp"]
        extra_link_args.append("-fopenmp")
        if sys.platform == "darwin":
            configure_macos_openmp(extra_compile_args, extra_link_args)
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib):
                extra_link_args.extend(
                    [
                        f"-L{torch_lib}",
                        f"-Wl,-rpath,{torch_lib}",
                    ]
                )

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


ext_modules = get_extensions()

if not ext_modules:
    setup()
else:
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
