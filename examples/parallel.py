"""
An example on how to efficiently create descriptors by using multiple
processes.
"""
import multiprocessing

import numpy as np

import ase.io

from describe.descriptors import CoulombMatrix
import describe.utils


def create(data):
    """This is the function that is called by each process but with different
    parts of the data -> data parallellism
    """
    cm = CoulombMatrix(n_atoms_max)
    n_samples = len(data)
    n_features = cm.get_number_of_features()
    inputs = np.empty((n_samples, n_features))

    # Create descriptors for the dataset
    for i_sample, sample in enumerate(data):

        # Set charges and Create Coulomb matrix
        sample.set_initial_charges(sample.get_atomic_numbers())
        inputs[i_sample, :] = cm.create(sample)

    # Return the list of features for each sample
    return inputs


if __name__ == '__main__':

    # Open a data file containing multiple configurations
    atoms_list = ase.io.iread("multiple.extxyz")
    atoms_list = list(atoms_list)

    # Split the data into roughly equivalent chunks for each process
    n_proc = 4
    k, m = divmod(len(atoms_list), n_proc)
    atoms_split = (atoms_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_proc))

    # Find out the maximum number of atoms in the data. This variable is shared
    # to all the processes in the create function
    n_atoms_max = describe.utils.system_stats(atoms_list)["n_atoms_max"]

    # Initialize a pool of processes, and tell each process in the pool to
    # handle a different part of the data
    pool = multiprocessing.Pool(processes=n_proc)
    results = pool.map(create, atoms_split)

    # Combine the results at the end when all processes have finished
    results = np.concatenate(results)
