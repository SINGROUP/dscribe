import scipy.sparse


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
