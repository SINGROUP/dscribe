"""
An example on how to parallelly create descriptors by using the python
multiprocessing package and the batch_create method.
"""
import ase.build.bulk

from dscribe.descriptors import CoulombMatrix
import dscribe.utils.stats
import dscribe.utils.batch_create


# Define a dataset
data = [
    ase.build.bulk("NaCl", "rocksalt", 5.64),
    ase.build.bulk("C", "diamond", 3.567),
    ase.build.bulk("Al", "fcc", 4.046),
    ase.build.bulk("GaAs", "zincblende", 5.653),
]

# Setup a descriptor
stats = dscribe.utils.system_stats(data)
n_max = stats["n_atoms_max"]
cm = CoulombMatrix(n_atoms_max=n_max)

# Create desciptors for the whole dataset
output = dscribe.utils.batch_create(cm, data, n_proc=2)
