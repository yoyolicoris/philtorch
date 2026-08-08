from setuptools import setup
import os
import glob
import sys
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
        if sys.platform == "darwin":
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
