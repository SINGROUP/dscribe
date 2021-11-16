<img src="https://raw.githubusercontent.com/SINGROUP/dscribe/master/logo/dscribe_logo.png" width="400">

[![Build Status](https://dev.azure.com/laurihimanen/DScribe%20CI/_apis/build/status/SINGROUP.dscribe?branchName=master)](https://dev.azure.com/laurihimanen/DScribe%20CI/_build/latest?definitionId=1&branchName=master)
[![Coverage Status](https://coveralls.io/repos/github/SINGROUP/dscribe/badge.svg?branch=master)](https://coveralls.io/github/SINGROUP/dscribe?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DScribe is a Python package for transforming atomic structures into fixed-size
numerical fingerprints. These fingerprints are often called "descriptors" and
they can be used in various tasks, including machine learning, visualization,
similarity analysis, etc.

# Homepage
For more details and tutorials, visit the homepage at:
[https://singroup.github.io/dscribe/](https://singroup.github.io/dscribe/)

# Quick Example
```python
import numpy as np
from ase.build import molecule
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix

# Define atomic structures
samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]

# Setup descriptors
cm_desc = CoulombMatrix(n_atoms_max=3, permutation="sorted_l2")
soap_desc = SOAP(species=["C", "H", "O", "N"], rcut=5, nmax=8, lmax=6, crossover=True)

# Create descriptors as numpy arrays or sparse arrays
water = samples[0]
coulomb_matrix = cm_desc.create(water)
soap = soap_desc.create(water, positions=[0])

# Easy to use also on multiple systems, can be parallelized across processes
coulomb_matrices = cm_desc.create(samples)
coulomb_matrices = cm_desc.create(samples, n_jobs=3)
oxygen_indices = [np.where(x.get_atomic_numbers() == 8)[0] for x in samples]
oxygen_soap = soap_desc.create(samples, oxygen_indices, n_jobs=3)

# Some descriptors also allow calculating derivatives with respect to atomic
# positions
der, des = soap_desc.derivatives(samples, method="auto", return_descriptor=True)
```

# Currently implemented descriptors
 | Descriptor                                    |  Spectrum |  Derivatives |
 |-----------------------------------------------|-----|-------|
 | Coulomb matrix                                | :heavy_check_mark: | :heavy_check_mark: |
 | Sine matrix                                   | :heavy_check_mark: | |
 | Ewald matrix                                  | :heavy_check_mark: | |
 | Atom-centered Symmetry Functions (ACSF)       | :heavy_check_mark: | |
 | Smooth Overlap of Atomic Positions (SOAP)     | :heavy_check_mark: |:heavy_check_mark: |
 | Many-body Tensor Representation (MBTR)        | :heavy_check_mark: | |
 | Local Many-body Tensor Representation (LMBTR) | :heavy_check_mark: | |
 | Valle-Oganov descriptor                       | :heavy_check_mark: | |

# Installation
In-depth installation instructions can be found [in the
documentation](https://singroup.github.io/dscribe/latest/install.html), but in
short:

## pip
```sh
pip install dscribe
```

## conda
```sh
conda install -c conda-forge dscribe
```

## From source
```sh
git clone https://github.com/SINGROUP/dscribe.git
cd dscribe
git submodule update --init
pip install .
```
