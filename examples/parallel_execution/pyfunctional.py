"""
Demonstrated how to parallelly create multiple descriptors for a dataset with
the PyFunctional library (http://www.pyfunctional.org/) that is similar to the
PySpark API.
"""
from functional.streams import ParallelStream as pseq

from collections import namedtuple

import ase.build.bulk

from describe.descriptors import CoulombMatrix
from describe.descriptors import SineMatrix
from describe.descriptors import EwaldMatrix

# Setup the descriptors
n_atoms_max = 4
n_proc = 4
coulombmatrix = CoulombMatrix(n_atoms_max=n_atoms_max)
sinematrix = SineMatrix(n_atoms_max=n_atoms_max)
ewaldmatrix = EwaldMatrix(n_atoms_max=n_atoms_max)

# Define a dataset
data = {
    "NaCl": ase.build.bulk("NaCl", "rocksalt", 5.64),
    "Diamond": ase.build.bulk("C", "diamond", 3.567),
    "Al": ase.build.bulk("Al", "fcc", 4.046),
    "GaAs": ase.build.bulk("GaAs", "zincblende", 5.653),
}

# Setup an iterable that runs through the samples.
Result = namedtuple("Result", "cm sm em")
Sample = namedtuple("Sample", "key value")
samples = [Sample(key, value) for key, value in data.items()]


def create_descriptors(atoms):
    """This function defines and sets up the descriptors that we are going to create.
    """
    cm = coulombmatrix.create(atoms)
    sm = sinematrix.create(atoms)
    em = ewaldmatrix.create(atoms, rcut=10, gcut=10)

    return Result(cm, sm, em)

# Use an operator chain similar to PySpark for creating the descriptors
res = (pseq(processes=n_proc)(samples)
    .filter(lambda x: len(x.value) <= n_atoms_max)         # Filters out systems that are too big
    .map(lambda x: [x.key, create_descriptors(x.value)])  # Create the descriptors
    .to_dict())                                           # Saves results as a dictionary
