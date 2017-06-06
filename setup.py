from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name='describe',
        version='0.1',
        description='Python package for creating machine learning descriptors for atomistic systems.',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'scipy',
            'ase',
        ],
    )
