import scipy.sparse


def get_adjacency_matrix(radius, pos1, pos2=None, output_type="coo_matrix"):
    """Calculates a sparse adjacency matrix by only considering distances
    within a certain cutoff. Uses a k-d tree to reach O(n log(N)) time
    complexity.

    Args:
        radius(float): The cutoff radius within which distances are
            calculated. Distances outside this radius are not included.
        pos1(np.ndarray): A list of N-dimensional positions.
        pos2(np.ndarray): A list of N-dimensional positions. If not provided,
            is assumed to be the same as pos1.
        output_type(str): Which container to use for output data. Options:
            "dok_matrix", "coo_matrix", "dict", or "ndarray". Default:
            "dok_matrix".

    Returns:
        dok_matrix | np.array | coo_matrix | dict: Symmetric sparse 2D
        matrix containing the pairwise distances.
    """
    tree1 = scipy.spatial.cKDTree(
        pos1,
        leafsize=16,
        compact_nodes=True,
        copy_data=False,
        balanced_tree=True,
        boxsize=None
    )
    if pos2 is None:
        pos2 = pos1
    tree2 = scipy.spatial.cKDTree(
        pos2,
        leafsize=16,
        compact_nodes=True,
        copy_data=False,
        balanced_tree=True,
        boxsize=None
    )
    dmat = tree1.sparse_distance_matrix(tree2, radius, output_type=output_type)

    return dmat


def get_adjacency_list(adjacency_matrix):
    """Used to transform an adjacency matrix into an adjacency list. The
    adjacency list provides much faster access to the neighbours of a node.

    Args:
        adjacency_matrix(scipy.sparse.spmatrix): The adjacency matrix from
            which the adjacency list is constructed from. Any of the scipy
            sparse matrix classes.

    Returns:
        list: A list of neighbouring indices. The list of neighbouring indices
        for atom at index i is given by accessing the ith element of this list.
    """
    # Ensure that we have a coo-matrix
    if type(adjacency_matrix) != scipy.sparse.coo_matrix:
        adjacency_matrix = adjacency_matrix.tocoo()

    # Build adjacency list
    adjacency_list = [[] for i in range(adjacency_matrix.shape[0])]
    for i, j in zip(adjacency_matrix.row, adjacency_matrix.col):
        adjacency_list[i].append(j)

    return adjacency_list
