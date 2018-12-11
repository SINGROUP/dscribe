# DScribe

[![Build Status](https://travis-ci.org/SINGROUP/dscribe.svg?branch=master)](https://travis-ci.org/SINGROUP/dscribe)
[![Coverage Status](https://coveralls.io/repos/github/SINGROUP/dscribe/badge.svg?branch=master)](https://coveralls.io/github/SINGROUP/dscribe?branch=master)

DScribe is a python package for creating machine learning descriptors for
atomistic systems. For more details and tutorials, visit the homepage at:
[https://singroup.github.io/dscribe/](https://singroup.github.io/dscribe/)

# Quick Example
```python
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix
from ase.build import molecule

# Define geometry
mol = molecule("H2O")

# Setup descriptors
cm_desc = CoulombMatrix(n_atoms_max=3, permutation="sorted_l2")
soap_desc = SOAP(atomic_numbers=[1, 8], rcut=5, nmax=8, lmax=6, crossover=True)

# Create descriptors as numpy arrays or scipy sparse matrices
input_cm = cm_desc.create(mol)
input_soap = soap_desc.create(mol, positions=[0])
```

# Currently implemented descriptors
 * Coulomb matrix
 * Sine matrix
 * Ewald matrix
 * Atom-centered Symmetry Functions (ACSF)
 * Smooth Overlap of Atomic Orbitals (SOAP)
 * Many-body Tensor Representation (MBTR)

# Installation
The package is compatible both with Python 2 and Python 3 (tested on 2.7 and
3.6). We currently only support Unix-based systems, including Linux and macOS.
The exact list of dependencies are given in setup.py and all of them will be
automatically installed during setup.

The latest stable release is available through pip: (add the -\-user flag if
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
