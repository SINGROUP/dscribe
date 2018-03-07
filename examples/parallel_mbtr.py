"""
An example on how to efficiently create the MBTR descriptor by using multiple
processes.
"""
import multiprocessing

import numpy as np

from describe.descriptors import MBTR
import describe.utils

import ase.io

from scipy.sparse import lil_matrix, save_npz


def create(data):
    """This is the function that is called by each process but with different
    parts of the data -> data parallellism
    """
    i_part = data[0]
    samples = data[1]
    mbtr = MBTR(
        atomic_numbers=atomic_numbers,
        k=[2],
        periodic=True,
        grid={
            "k1": {
                "min": min(atomic_numbers)-1,
                "max": max(atomic_numbers)+1,
                "sigma": 0.1,
                "n": 200,
            },
            "k2": {
                "min": 0,
                "max": 1/min_distance,
                "sigma": 0.01,
                "n": 200,
            },
        },
        weighting={
            "k2": {
                "function": lambda x: np.exp(-0.5*x),
                "threshold": 1e-3
            },
        },
        flatten=True,
    )
    n_samples = len(samples)
    n_features = int(mbtr.get_number_of_features())
    mbtr_inputs = lil_matrix((n_samples, n_features))

    # Create descriptors for the dataset
    for i_sample, system in enumerate(samples):
        # Set charges and create descriptors
        system.charges = system.numbers
        mbtr_mat = mbtr.create(system)
        mbtr_inputs[i_sample, :] = mbtr_mat

    # Return the list of features for each sample
    return {
        "part": i_part,
        "mbtr": mbtr_inputs,
    }


def split(items, n):
    """
    """
    k, m = divmod(len(items), n)
    splitted = (items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return splitted


if __name__ == '__main__':

    # Open a data file containing multiple configurations
    samples = ase.io.iread("multiple.extxyz")
    samples = list(samples)
    n_samples = len(samples)

    # Split the data into roughly equivalent chunks for each process. The entries
    n_proc = 4
    samples_split = split(samples, n_proc)
    id_samples_tuple = [(x[0], x[1]) for x in enumerate(samples_split)]

    # Find out the maximum number of atoms in the data. This variable is shared
    # to all the processes in the create function
    stats = describe.utils.system_stats(samples)
    atomic_numbers = stats["atomic_numbers"]
    n_atoms_max = stats["n_atoms_max"]
    min_distance = stats["min_distance"]

    # Initialize a pool of processes, and tell each process in the pool to
    # handle a different part of the data
    pool = multiprocessing.Pool(processes=n_proc)
    results = pool.map(create, id_samples_tuple)

    # Sort results to the original order
    results = sorted(results, key=lambda x: x["part"])
    n_features = int(results[0]["mbtr"].shape[1])

    # Combine the results at the end when all processes have finished
    mbtr_list = lil_matrix((n_samples, n_features))
    id_list = []
    i_id = 0
    for result in results:
        i_n_samples = result["mbtr"].shape[0]
        mbtr_in = result["mbtr"]
        mbtr_list[i_id:i_id+i_n_samples, :] = mbtr_in
        i_id += i_n_samples

    # Convert lil_matrix to CSR format. The lil format is good for creating a
    # sparse matrix, CSR is good for efficient math.
    mbtr_list = mbtr_list.tocsr()

    # Save results as a sparse matrix.
    save_npz(".mbtr.npz", mbtr_list)
