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
 * Coulomb matrix
 * Sine matrix
 * Ewald matrix
 * Atom-centered Symmetry Functions (ACSF)
 * Smooth Overlap of Atomic Positions (SOAP)
 * Many-body Tensor Representation (MBTR)
 * Local Many-body Tensor Representation (LMBTR)

# Installation
The newest versions of the package are compatible with Python >= 3.6 (tested on
3.6, 3.7 and 3.8). DScribe versions <= 0.2.7 also support Python 2.7. We
currently only support Unix-based systems, including Linux and macOS. For
Windows-machines we suggest using the [Windows Subsystem for Linux (WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).
The exact list of dependencies are given in setup.py and all of them will be
automatically installed during setup.

The package contains C++ extensions that are automatically compiled during
install. On Linux-based systems the compilation tools are typically installed
by default, on MacOS you may need to install additional command line tools if
facing issues during setup ([see common issues in the
documentation](https://singroup.github.io/dscribe/latest/install.html).

The latest stable release is available through pip: (add the `--user` flag if
root access is not available)

```sh
pip install dscribe
```

To install the latest development version, clone the source code from github
and install with pip from local file:

```sh
git clone https://github.com/SINGROUP/dscribe.git
cd dscribe
pip install .
```
