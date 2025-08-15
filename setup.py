from setuptools import setup
import os
import glob
import torch
import re
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "philtorch"


def format_branch_name(name):
    # "(fix|feat)/issue-name"
    pattern = re.compile("^(fix|feat)\/(?P<branch>.+)")

    match = pattern.search(name)
    if match:
        return f"dev+{match.group(0)}"  # => dev+"(fix|feat)/issue-name"

    # function is called even if branch name is not used in a current template
    # just left properly named branches intact
    if name in ["master", "dev"]:
        return name

    # fail in case of wrong branch names like "bugfix/issue-unknown"
    raise ValueError(f"Wrong branch name: {name}")


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
