from setuptools import setup
from pathlib import Path

root_path = Path(__file__).parent
version_file = root_path / "VERSION.txt"

with open("README.md", "r") as fh:
    long_description = fh.read()

author = "Chin-Yun Yu"
author_email = "chin-yun.yu@qmul.ac.uk"

setup(
    name="philtorch",
    version="0.1.0",
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description="A PyTorch library for time-varying IIR filters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyolicoris/philtorch",
    license="MIT",
    license_files=["LICENSE"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=["philtorch"],
    include_package_data=True,
)
