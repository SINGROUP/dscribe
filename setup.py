import platform
from setuptools import setup, find_packages, Extension

# On mac we need to specify that a minimum target version to get the correct
# c++11 compilation flags (libc++) in place
cpp_extra_compile_args = ['-std=c++11']
if platform.system() == "Darwin":
    cpp_extra_compile_args.append('-mmacosx-version-min=10.9')

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
        version="0.2.3",
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
