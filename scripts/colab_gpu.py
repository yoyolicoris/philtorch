"""Run philtorch's CUDA tests, or one benchmark, on a Google Colab GPU.

A local driver for the ``colab`` CLI. It uploads a CUDA wheel and the working
tree's ``tests/``, ``benchmarks/`` and ``pyproject.toml`` to a Colab VM, installs
the wheel next to the newest PyTorch of ``torch_matrix.json`` that has the
wheel's CUDA major, and runs there, in a directory without the package's source,
what is asked::

    python scripts/colab_gpu.py test --gpu A100 --wheel wheel/philtorch-...+cu13-....whl [--stop]
    python scripts/colab_gpu.py bench --gpu A100 --wheel wheel/....whl [--stop] \\
        benchmarks/benchmark_recur.py [script args...]

The session is named ``philtorch-<gpu>`` and reused when it exists. A kept
session bills until it is stopped: pass ``--stop``, or run ``colab stop``.
The exit code is non-zero when the remote run failed.
"""

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from torch_matrix import INDEX_URL, REPO_ROOT, cuda_major, version_tuple  # noqa: E402

GPUS = ("T4", "L4", "A100", "G4", "H100")
# Printed by the remote code when it ends well: `colab exec` exits with 0 after
# a remote exception, whose traceback it only prints.
SENTINEL = "COLAB_GPU_OK"

# Runs in the Colab kernel, after a prelude that sets MODE, TORCH, INDEX, WHEEL,
# WHEEL_VERSION, TARBALL, BENCH and SENTINEL.
REMOTE = r'''
import importlib.metadata
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/tmp/philtorch-work")
CUDA_MARK = re.compile(r"(@|marks=)pytest\.mark\.cuda\b")


def run(*cmd, cwd=None):
    # The Colab kernel captures this process's prints only, not a child's
    # stdout, so the child's output is relayed line by line.
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, cmd)


run("nvidia-smi")

pip = [sys.executable, "-m", "pip", "install", "-q"]
try:
    have_torch = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError:
    have_torch = ""
# The index's wheels carry its CUDA build as their local version: 2.15.0+cu134.
if have_torch != f"{TORCH}+{INDEX.rsplit('/', 1)[1]}":
    run(*pip, f"torch=={TORCH}", "--index-url", INDEX)
run(*pip, "pytest", "pytest-cov", "numpy", "scipy")
run(*pip, "--no-deps", "--force-reinstall", WHEEL)

# Only tests/, benchmarks/ and pyproject.toml, so that no source of philtorch is
# importable from the working directory.
shutil.rmtree(WORK, ignore_errors=True)
with tarfile.open(TARBALL) as tar:
    tar.extractall(WORK, filter="data")

check = """
import sys
from pathlib import Path

import torch
import philtorch

major, minor = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on "
      f"{torch.cuda.get_device_name()}, sm_{major}{minor}")
print(f"philtorch {philtorch.__version__} from {philtorch.__file__}")
assert philtorch.__version__ == sys.argv[1], (philtorch.__version__, sys.argv[1])
assert not Path(philtorch.__file__).is_relative_to(Path.cwd()), "imported from the work dir"
"""
run(sys.executable, "-c", check, WHEEL_VERSION, cwd=WORK)

if MODE == "test":
    modules = sorted(
        str(path.relative_to(WORK))
        for path in (WORK / "tests").glob("test_*.py")
        if CUDA_MARK.search(path.read_text())
    )
    run(
        sys.executable, "-m", "pytest", *modules,
        "-m", "cuda", "-p", "no:cacheprovider", "-q", "-ra", "--no-cov",
        cwd=WORK,
    )
else:
    run(sys.executable, *BENCH, cwd=WORK)
print(SENTINEL, flush=True)
'''


def wheel_version(wheel):
    """The version of a wheel file named ``{name}-{version}-{tags}.whl``."""
    return wheel.name.split("-")[1]


def pick_torch(version, matrix):
    """The newest torch of the matrix with a CUDA of the wheel's major, and its index URL.

    The index is that of the newest CUDA of that major. ``version`` is the wheel's,
    which ends in ``+cu<major>``.
    """
    match = re.search(r"\+cu(\d+)$", version)
    if not match:
        raise SystemExit(f"{version}: not a CUDA wheel version, which ends in +cu<major>")
    major = match[1]
    for entry in reversed(matrix["torch"]):
        cudas = [cuda for cuda in entry["cuda"] if cuda_major(cuda) == major]
        if cudas:
            cuda = max(cudas, key=version_tuple)
            return entry["version"], INDEX_URL + "cu" + cuda.replace(".", "")
    raise SystemExit(f"torch_matrix.json has no torch with CUDA {major}")


def pack(path):
    """Tar ``tests/``, ``benchmarks/`` and ``pyproject.toml``, without ``__pycache__``."""

    def keep(info):
        return None if "__pycache__" in Path(info.name).parts else info

    with tarfile.open(path, "w:gz") as tar:
        for name in ("tests", "benchmarks", "pyproject.toml"):
            tar.add(REPO_ROOT / name, arcname=name, filter=keep)


def colab(*args):
    subprocess.run(["colab", *args], check=True)


def session_exists(name):
    """Whether ``colab sessions`` lists a session called ``name``."""
    out = subprocess.run(["colab", "sessions"], check=True, capture_output=True, text=True).stdout
    return name in re.findall(r"^\[(.*?)\] ", out, flags=re.MULTILINE)


def run_remote(name, source):
    """Stream the remote run's output; whether it ended with the sentinel and exit code 0."""
    proc = subprocess.Popen(
        # No limit on the time without output: pip and the tests can be silent for minutes.
        ["colab", "exec", "-s", name, "--timeout", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc.stdin.write(source)
    proc.stdin.close()
    done = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        done = done or line.strip() == SENTINEL
    return proc.wait() == 0 and done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode, help in (("test", "run the CUDA tests"), ("bench", "run one benchmark script")):
        sub = commands.add_parser(mode, help=help)
        sub.add_argument("--gpu", required=True, choices=GPUS)
        sub.add_argument("--wheel", required=True, type=Path, help="the CUDA wheel to install")
        sub.add_argument("--stop", action="store_true", help="stop the session afterwards")
        if mode == "bench":
            sub.add_argument("script", help="a script of the working tree, e.g. benchmarks/x.py")
            sub.add_argument("args", nargs=argparse.REMAINDER, help="arguments of the script")
    args = parser.parse_args(argv)

    if not args.wheel.is_file():
        parser.error(f"{args.wheel}: no such file")
    if args.mode == "bench" and not (REPO_ROOT / args.script).is_file():
        parser.error(f"{args.script}: no such file in {REPO_ROOT}")
    matrix = json.loads((REPO_ROOT / "torch_matrix.json").read_text(encoding="utf-8"))
    version = wheel_version(args.wheel)
    torch, index = pick_torch(version, matrix)

    name = f"philtorch-{args.gpu.lower()}"
    if not session_exists(name):
        colab("new", "-s", name, "--gpu", args.gpu)
    ok = False
    try:
        remote_wheel = f"/tmp/{args.wheel.name}"
        remote_tar = "/tmp/philtorch-tree.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            tarball = Path(tmp) / "tree.tar.gz"
            pack(tarball)
            colab("upload", "-s", name, str(tarball), remote_tar)
        colab("upload", "-s", name, str(args.wheel), remote_wheel)
        values = {
            "MODE": args.mode,
            "TORCH": torch,
            "INDEX": index,
            "WHEEL": remote_wheel,
            "WHEEL_VERSION": version,
            "TARBALL": remote_tar,
            "BENCH": [args.script, *args.args] if args.mode == "bench" else [],
            "SENTINEL": SENTINEL,
        }
        prelude = "".join(f"{key} = {value!r}\n" for key, value in values.items())
        ok = run_remote(name, prelude + REMOTE)
    finally:
        if args.stop:
            colab("stop", "-s", name)
        else:
            print(
                f"Session {name} is still running and billing; stop it with: colab stop -s {name}"
            )
    if not ok:
        print("The remote run failed.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
