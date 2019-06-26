import os
import platform
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension

cpp_extra_compile_args = ['-std=c++11']

# Taken from the setup.py file in the pandas-library, see the following issue
# for more details: https://github.com/pandas-dev/pandas/issues/23424. For mac,
# ensure extensions are built for macos 10.9 when compiling on a 10.9 system or
# above, overriding distutils behaviour which is to target the version that
# python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if platform.system() == "Darwin":
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

extensions = [
    # The ACSF C++ extension, wrapped with cython
    Extension(
        "dscribe.libacsf.acsfwrapper",
        [
            "dscribe/libacsf/acsfwrapper.cpp",
        ],
        language='c++',
        include_dirs=["dscribe/libacsf"],
        extra_compile_args=cpp_extra_compile_args,
    ),
    # The MBTR C++ extension, wrapped with cython
    Extension(
        "dscribe.libmbtr.mbtrwrapper",
        [
            "dscribe/libmbtr/mbtrwrapper.cpp",
        ],
        language='c++',
        include_dirs=["dscribe/libmbtr"],
        extra_compile_args=cpp_extra_compile_args,
    ),
]

if __name__ == "__main__":
    setup(name='dscribe',
        version="0.2.4",
        url="https://singroup.github.io/dscribe/",
        description='A Python package for creating feature transformations in applications of machine learning to materials science.',
        long_description='A Python package for creating feature transformations in applications of machine learning to materials science.',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'scipy',
            'ase',
            'future',
            'scikit-learn',
            'joblib',
            'soaplite==1.0.3',
        ],
        include_package_data=True,  # This ensures that files defined in MANIFEST.in are included
        ext_modules=extensions,
        license="Apache License 2.0",
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Physics',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.2',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        keywords='descriptor machine learning atomistic structure materials science',
        python_requires='>=2.6, <4',
    )
