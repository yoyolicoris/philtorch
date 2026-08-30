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

    torchlpc_root = os.path.join(
        this_dir, "third_party", "torchlpc", "torchlpc", "csrc"
    )
    torchlpc_sources = [
        os.path.join(torchlpc_root, "cuda", name)
        for name in ("lpc.cu", "linear_recurrence.cu")
    ]
    pararnn_root = os.path.join(this_dir, "third_party", "pararnn", "pararnn", "csrc")
    pararnn_sources = [os.path.join(pararnn_root, "parallel_reduce.cu")]

    if use_cuda:
        vendored_sources = [*torchlpc_sources, *pararnn_sources]
        missing_sources = [path for path in vendored_sources if not os.path.isfile(path)]
        if missing_sources:
            relative_paths = [os.path.relpath(path, this_dir) for path in missing_sources]
            raise RuntimeError(
                "CUDA builds require initialized third-party submodules. Run "
                "'git submodule update --init --recursive'. Missing: "
                + ", ".join(relative_paths)
            )

        extra_compile_args.setdefault("cxx", []).append(f"-I{torchlpc_root}")
        extra_compile_args.setdefault("cxx", []).append(
            f"-I{os.path.join(torchlpc_root, 'cuda')}"
        )
        extra_compile_args.setdefault("cxx", []).append(f"-I{pararnn_root}")
        extra_compile_args.setdefault("cxx", []).extend(
            ["-DFLOAT64_CHUNK_SIZE_DIAG=4", "-DFLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=1"]
        )
        extra_compile_args.setdefault("nvcc", []).extend(
            [
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-D__CUDA_INCLUDE_HALF_OPERATORS__",
                "-D__CUDA_INCLUDE_BFLOAT16_OPERATORS__",
                "-Xcudafe",
                "--diag_suppress=1886",
                "-DFLOAT64_CHUNK_SIZE_DIAG=4",
                "-DFLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=1",
            ]
        )

    if sys.platform == "darwin":
        sources += list(glob.glob(os.path.join(extensions_dir, "*.mm")))
        extra_link_args.extend(["-framework", "Foundation", "-framework", "Metal"])

    if not use_cuda:
        sources = [s for s in sources if "pararnn_shim" not in os.path.basename(s)]

    if use_cuda:
        sources += cuda_sources
        sources += torchlpc_sources
        sources += pararnn_sources
        extra_compile_args.setdefault("nvcc", []).append("--extended-lambda")

    if len(sources) == 0:
        return []

    ext_modules = [
        extension(
            f"{library_name}._C",
            [os.path.relpath(s, this_dir) for s in sources],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


try:
    ext_modules = get_extensions()
except ImportError:
    # Only torch's absence is treated as "maybe metadata-only"; other errors
    # from get_extensions() (e.g. missing Homebrew on macOS) propagate as-is,
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

data_files = [
    (
        "share/doc/philtorch",
        [
            "NOTICE",
            "LICENSES/README.md",
            "LICENSES/torchlpc-LICENSE",
            "LICENSES/pararnn-LICENSE",
        ],
    )
]

if not ext_modules:
    setup(data_files=data_files)
else:
    from torch.utils.cpp_extension import BuildExtension

    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        data_files=data_files,
    )
