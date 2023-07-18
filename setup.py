import sys

# Check python version
if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

import platform
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
from setuptools import setup, find_packages, Extension
from subprocess import getoutput


def using_clang():
    """Will we be using a clang compiler?
    """
    compiler = new_compiler()
    customize_compiler(compiler)
    compiler_ver = getoutput("{0} -v".format(compiler.compiler[0]))
    return "clang" in compiler_ver


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


cpp_extra_link_args = []
cpp_extra_compile_args = [
    "-std=c++11",                   # C++11
    "-O3",                          # O3 optimizations
    "-I dependencies/eigen/Eigen/"  # Eigen dependency
]

# Needed to specify C++ runtime library on OSX. This solution is replicated
# from the setup.py of mdanalysis
if platform.system() == "Darwin" and using_clang():
    cpp_extra_compile_args.append("-stdlib=libc++")
    cpp_extra_compile_args.append("-mmacosx-version-min=10.9")
    cpp_extra_link_args.append("-stdlib=libc++")
    cpp_extra_link_args.append("-mmacosx-version-min=10.7")

extensions = [
    # The SOAP, MBTR, ACSF and utils C++ extensions, wrapped with pybind11
    Extension(
        'dscribe.ext',
        [
            "dscribe/ext/ext.cpp",
            "dscribe/ext/celllist.cpp",
            "dscribe/ext/descriptorglobal.cpp",
            "dscribe/ext/descriptor.cpp",
            "dscribe/ext/cm.cpp",
            "dscribe/ext/soap.cpp",
            "dscribe/ext/soapGTO.cpp",
            "dscribe/ext/soapGeneral.cpp",
            "dscribe/ext/acsf.cpp",
            "dscribe/ext/mbtr.cpp",
            "dscribe/ext/geometry.cpp",
            "dscribe/ext/weighting.cpp",
        ],
        include_dirs=[
            # Path to Eigen headers
            "dependencies/eigen",
            # Path to pybind11 headers
            "dscribe/ext",
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=cpp_extra_compile_args + ["-fvisibility=hidden"],  # the -fvisibility flag is needed by pybind11
        extra_link_args=cpp_extra_link_args,
    )
]

if __name__ == "__main__":
    setup(
        name="dscribe",
        version="2.0.1",
        url="https://singroup.github.io/dscribe/",
        description="A Python package for creating feature transformations in applications of machine learning to materials science.",
        long_description="A Python package for creating feature transformations in applications of machine learning to materials science.",
        packages=find_packages(),
        setup_requires=['pybind11>=2.4'],
        install_requires=['pybind11>=2.4', "numpy", "scipy", "ase>=3.19.0", "scikit-learn", "joblib>=1.0.0", "sparse"],
        include_package_data=True,  # This ensures that files defined in MANIFEST.in are included
        ext_modules=extensions,
        license="Apache License 2.0",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Physics",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS",
            "Operating System :: Unix",
            "Programming Language :: C",
            "Programming Language :: C++",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3 :: Only",
        ],
        keywords="descriptor machine learning atomistic structure materials science",
        python_requires=">=3.7",
    )
