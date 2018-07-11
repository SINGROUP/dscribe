from setuptools import setup, find_packages, Extension

extensions = [
    Extension(
        "describe.libacsf.libacsf",
        [
            "describe/libacsf/acsf-utils.c",
            "describe/libacsf/acsf-compute.c",
            #"describe/libacsf/acsf.c"
        ],
        include_dirs=["describe/libacsf"],
        libraries=["m"],
        extra_compile_args=["-O3", "-std=c99"]
    )
]

if __name__ == "__main__":
    setup(name='describe',
        version='0.1',
        url="https://singroup.github.io/describe/",
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
        package_data={'describe': ['describe/libacsf/acsf.h']},
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
        keywords='atoms structure materials science crystal symmetry',
        python_requires='>=2.6, <4',
    )
