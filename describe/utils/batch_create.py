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
    verbose = inp[2]
    proc_id = inp[3]

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

    for i_sample, sample in enumerate(samples):
        vec = descriptor.create(sample)

        if is_sparse:
            data.append(vec.data)
            rows.append(vec.rows + i_sample)
            cols.append(vec.cols)
        else:
            results[i_sample, :] = vec

        if verbose:
            if i_sample % int(n_samples/100) == 0:
                print("Process {0}: {1:.1f} %".format(proc_id, (i_sample+1)/n_samples*100))

    if is_sparse:
        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        results = coo_matrix((data, (rows, cols)), shape=[n_samples, n_features], dtype=np.float32)

    return results


def batch_create(descriptor, samples, n_proc, verbose=True):
    """Used to create a descriptor output for multiple samples in parallel and
    store the result in a n_samples x n_features sparse or dense array.

    Uses the python multiprocessing library and data parallellism to create the
    descriptors in parallel.

    Args:
        samples:
        n_proc:
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
    inputs = [(x, descriptor, verbose, proc_id) for proc_id, x in enumerate(atoms_split)]

    # Initialize a pool of processes, and tell each process in the pool to
    # handle a different part of the data
    pool = multiprocessing.Pool(processes=n_proc)
    vec_lists = pool.map(create, inputs)  # pool.map keeps the order

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
