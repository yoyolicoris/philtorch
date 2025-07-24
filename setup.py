from setuptools import setup
import os
import glob
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "philtorch"


def get_extensions():
    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    use_openmp = torch.backends.openmp.is_available()
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {}
    if use_openmp:
        extra_compile_args["cxx"] = ["-fopenmp"]
        extra_link_args.append("-fopenmp")

    this_dir = os.path.abspath(os.path.dirname(__file__))
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources
        extra_compile_args["nvcc"] = ["--extended-lambda"]

    ext_modules = [
        extension(
            f"{library_name}._C",
            # sources,
            # [os.path.relpath(s, this_dir) for s in sources],
            ["philtorch/csrc/recur2.cu"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
