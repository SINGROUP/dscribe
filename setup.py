from setuptools import setup, find_packages, Extension

extensions = [Extension("describe.libacsf.libacsf",
                       ["describe/libacsf/acsf-utils.c", "describe/libacsf/acsf-compute.c", "describe/libacsf/acsf.c"],
                       include_dirs=["describe/libacsf"],
                       libraries=["m"],
                       extra_compile_args=["-O3", "-std=c99"]
              )]


if __name__ == "__main__":
    setup(name='describe',
        version='0.1',
        description='Python package for creating machine learning descriptors for atomistic systems.',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'scipy',
            'ase',
            'future',
            'matplotlib',
        ],
        package_data={'describe':['describe/libacsf/acsf.h']},
        ext_modules=extensions,
    )
