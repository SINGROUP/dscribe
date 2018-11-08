from setuptools import setup, find_packages, Extension

extensions = [
    Extension(
        "dscribe.libacsf.libacsf",
        [
            "dscribe/libacsf/acsf-utils.c",
            "dscribe/libacsf/acsf-compute.c",
        ],
        include_dirs=["dscribe/libacsf"],
        libraries=["m"],
        extra_compile_args=["-O3", "-std=c99"]
    ),
    Extension(
        "dscribe.libmbtr.cmbtrwrapper",
        [
            "dscribe/libmbtr/cmbtrwrapper.cpp",
        ],
        extra_compile_args=['-std=c++11'],
        include_dirs=["dscribe/libmbtr"],
    ),
]

if __name__ == "__main__":
    setup(name='dscribe',
        version='0.1.1',
        url="https://singroup.github.io/dscribe/",
        description='A Python package for creating feature transformations in applications of machine learning to materials science.',
        long_description='A Python package for creating feature transformations in applications of machine learning to materials science.',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'scipy',
            'ase',
            'future',
            'matplotlib',
            'soaplite',
        ],
        package_data={'dscribe': ['dscribe/libacsf/acsf.h']},
        ext_modules=extensions,
        license="Apache 2.0",
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
