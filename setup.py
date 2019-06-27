import sys
import platform
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
from setuptools import setup, find_packages, Extension
if sys.version_info[0] >= 3:
    from subprocess import getoutput
else:
    from commands import getoutput


def using_clang():
    """Will we be using a clang compiler?
    """
    compiler = new_compiler()
    customize_compiler(compiler)
    compiler_ver = getoutput("{0} -v".format(compiler.compiler[0]))
    return 'clang' in compiler_ver

cpp_extra_link_args = []
cpp_extra_compile_args = ['-std=c++11']

# Needed to specify X++ runtime library on OSX. This solution is replicated
# from the setup.py of mdanalysis
if platform.system() == 'Darwin' and using_clang():
    cpp_extra_compile_args.append('-stdlib=libc++')
    cpp_extra_compile_args.append('-mmacosx-version-min=10.9')
    cpp_extra_link_args.append('-stdlib=libc++')
    cpp_extra_link_args.append('-mmacosx-version-min=10.7')

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
        extra_link_args=cpp_extra_link_args,
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
        extra_link_args=cpp_extra_link_args,
    ),
]

if __name__ == "__main__":
    setup(name='dscribe',
        version="0.2.7",
        url="https://singroup.github.io/dscribe/",
        description='A Python package for creating feature transformations in applications of machine learning to materials science.',
        long_description='A Python package for creating feature transformations in applications of machine learning to materials science.',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'scipy',
            'ase',
            'future',
            'scikit-learn==0.20.3',
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
