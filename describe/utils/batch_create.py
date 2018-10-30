import multiprocessing

import numpy as np

from scipy.sparse import coo_matrix


def create(inp):
    """This is the function that is called by each process but with
    different parts of the data. This function is a module level function
    (instead of nested within batch_create), because only top level functions
    are picklable by the multiprocessing library.
    """
    samples = inp[0]
    descriptor = inp[1]
    pos = inp[2]
    verbose = inp[3]
    proc_id = inp[4]

    # Create descriptors for the dataset
    n_samples = len(samples)
    is_sparse = descriptor.sparse
    n_features = descriptor.get_number_of_features()

    if is_sparse:
        data = []
        rows = []
        cols = []
    else:
        results = np.empty((n_samples, n_features))

    old_percent = 0
    for i_sample, sample in enumerate(samples):
        if pos is None:
            vec = descriptor.create(sample)
        else:
            i_pos = pos[i_sample]
            vec = descriptor.create(sample, i_pos)

        if is_sparse:
            data.append(vec.data)
            rows.append(vec.row + i_sample)
            cols.append(vec.col)
        else:
            results[i_sample, :] = vec

        if verbose:
            current_percent = (i_sample+1)/n_samples*100
            if current_percent >= old_percent + 1:
                old_percent = current_percent
                print("Process {0}: {1:.1f} %".format(proc_id, current_percent))

    if is_sparse:
        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        results = coo_matrix((data, (rows, cols)), shape=[n_samples, n_features], dtype=np.float32)

    return results


def batch_create(descriptor, samples, n_proc, positions=None, create_func=None, verbose=True):
    """Used to create a descriptor output for multiple samples in parallel and
    store the result in a n_samples x n_features sparse or dense array.

    Uses the python multiprocessing library and data parallellism to create the
    descriptors in parallel.

    Args:
        samples:
        n_proc: The number of processes. The data will be split into this many
            parts and divided into different processes.
        positions (iterable): Needs to be specified if the given descriptor is
            local and requires a 'positions'-argument in the create-function.
            Should be a list of positions matching the given 'samples'.
        create_func (function): A custom function for creating the output from
            each process. If none specified a default function will be used.
            Takes in one tuple argument 'inp' with the following information:
                inp[0]: samples
                inp[1]: descriptor
                inp[2]: verbose parameter
                inp[3]: process id number
            The function should return a 2D array. If descriptor.sparse is set
            to true, the output should be a scipy.linalg.coo_matrix, otherwise
            a numpy.ndarray should be returned.
        verbose (boolean): Whether to report a percentage of the samples
            covered from each process.

    Returns:
        np.ndarray | scipy.sparse.coo_matrix: The descriptor vectors for all
        samples in a single (n_samples x n_features) array.
    """
    # Get number of samples and whether the output is sparse or not.
    n_features = descriptor.get_number_of_features()
    is_sparse = descriptor.sparse

    # If the descriptor is not flattened, the batch processing cannot be by
    # this function.
    flatten = descriptor.flatten
    if not flatten:
        raise ValueError(
            "The given descriptor is not specified to have flattened output "
            "with the 'flatten' constructor argument. Cannot save the "
            "descriptor output in a batch."
        )

    # Split the data into roughly equivalent chunks for each process
    k, m = divmod(len(samples), n_proc)
    atoms_split = (samples[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_proc))
    if positions is not None:
        positions_split = (positions[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_proc))
        inputs = [(x, descriptor, pos, verbose, proc_id) for proc_id, (x, pos) in enumerate(zip(atoms_split, positions_split))]
    else:
        inputs = [(x, descriptor, None, verbose, proc_id) for proc_id, x in enumerate(atoms_split)]

    # Initialize a pool of processes, and tell each process in the pool to
    # handle a different part of the data
    pool = multiprocessing.Pool(processes=n_proc)
    if create_func is None:
        create_func = create
    vec_lists = pool.map(create_func, inputs)  # pool.map keeps the order

    if is_sparse:
        # Put results into one big sparse matrix
        n_samples = len(samples)
        row_offset = 0
        data = []
        cols = []
        rows = []
        for i, i_res in enumerate(vec_lists):
            i_res = i_res.tocoo()
            i_n_samples = i_res.shape[0]
            i_data = i_res.data
            i_col = i_res.col
            i_row = i_res.row

            data.append(i_data)
            rows.append(i_row + row_offset)
            cols.append(i_col)

            # Increase the row offset
            row_offset += i_n_samples

        # Saves the descriptors as a sparse matrix
        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        results = coo_matrix((data, (rows, cols)), shape=[n_samples, n_features], dtype=np.float32)
        results = results.tocsr()
    else:
        results = np.concatenate(vec_lists, axis=0)

    return results
