from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, naive_recompile, build_ext

ext_modules = [
    Pybind11Extension(
        "dscribe.ext",
        sorted(glob("dscribe/ext/*.cpp")),
        include_dirs=[
            # Path to Eigen headers
            "dependencies/eigen",
            # Path to own headers
            "dscribe/ext",
        ],
        cxx_std=11,
        extra_compile_args=[
            "-O3",                          # O3 optimizations
            "-I dependencies/eigen/Eigen/", # Eigen dependency
        ],
    ),
]

with ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile):
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        python_requires=">=3.9",
    )
