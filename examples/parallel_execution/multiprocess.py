"""
An example on how to parallelly create descriptors by using the python
multiprocessing package.
"""
import multiprocessing

from collections import namedtuple

import ase.build.bulk

from describe.descriptors import CoulombMatrix
from describe.descriptors import EwaldMatrix
from describe.descriptors import SineMatrix


def create(data):
    """This is the function that is called by each process but with different
    parts of the data.
    """
    # Create descriptors for the dataset
    results = {}
    for key, value in data:
        cm = coulombmatrix.create(value)
        sm = sinematrix.create(value)
        em = ewaldmatrix.create(value, rcut=10, gcut=10)
        results[key] = Result(cm, sm, em)

    # Return the list of features for each sample
    return results

if __name__ == '__main__':

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

    # Split the data into roughly equivalent chunks for each process
    n_atoms_max = 4
    n_proc = 4
    coulombmatrix = CoulombMatrix(n_atoms_max=n_atoms_max)
    sinematrix = SineMatrix(n_atoms_max=n_atoms_max)
    ewaldmatrix = EwaldMatrix(n_atoms_max=n_atoms_max)
    k, m = divmod(len(samples), n_proc)
    atoms_split = (samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_proc))

    # Initialize a pool of processes, and tell each process in the pool to
    # handle a different part of the data
    pool = multiprocessing.Pool(processes=n_proc)
    results = pool.map(create, atoms_split)

    # The result dictionaries are combined here, as results now contains a list
    # of dictionaries, one for each process
    final_results = {}
    for dictionary in results:
        final_results.update(dictionary)
