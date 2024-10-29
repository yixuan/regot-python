# https://stackoverflow.com/a/47651621

import os
from pathlib import Path
from glob import glob
import zipfile
from pybind11.setup_helpers import Pybind11Extension, build_ext
import requests
import cpufeature
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

# The directory that contains setup.py
SETUP_DIRECTORY = Path(__file__).resolve().parent

# Directory of LBFGS++
LBFGSPP_DIRECTORY = f"{SETUP_DIRECTORY}/src/LBFGSpp"

# Download Eigen source files
# Modified from https://github.com/tohtsky/irspack/blob/main/setup.py
class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    EIGEN3_DIRNAME = "eigen-3.4.0"

    def __str__(self) -> str:
        # Test whether the environment variable EIGEN3_INCLUDE_DIR is set
        # If yes, directly return this directory
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)
        if eigen_include_dir is not None:
            return eigen_include_dir

        # If the directory already exists (e.g. from previous setup),
        # directly return it
        target_dir = SETUP_DIRECTORY / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return target_dir.name

        # Filename for the downloaded Eigen source package
        download_target_dir = SETUP_DIRECTORY / "eigen3.zip"
        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)
        # Unzip package
        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir.name

# Test CPU feature
# Better performance if AVX2 is supported and OpenMP is enabled
extra_compiler_args = []
if cpufeature.CPUFeature["num_virtual_cores"] > 1:
    extra_compiler_args += ["-fopenmp"]
if cpufeature.CPUFeature["AVX2"]:
    extra_compiler_args += ["-mavx2"]

ext_modules = [
    Pybind11Extension("regot._internal",
        sorted(glob("src/*.cpp")),
        include_dirs=[get_eigen_include(), LBFGSPP_DIRECTORY],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args = extra_compiler_args,
        extra_link_args = extra_compiler_args
    )
]

setup(
    name="regot",
    version=__version__,
    author="Yixuan Qiu",
    author_email="yixuanq@gmail.com",
    url="https://github.com/yixuan/regot-python",
    description="Regularized Optimal Transport",
    long_description="A collection of state-of-the-art solvers for regularized optimal transport, implemented in efficient C++ code.",
    packages=["regot"],
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
