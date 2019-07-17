from collections import defaultdict


def get_adjacency_list(adjacency_matrix, return_values=False):
    """Used to transform an adjacency matrix into an adjacency list. The
    adjacency list provides much faster access to the neighbours of a node.

    Args:

    Returns:
        dict: A dictionary with node index as key, and the neighbour indices as
        values.
    """
    coo = adjacency_matrix.tocoo()
    adjacency_list = defaultdict(list)
    if return_values is False:
        for i, j in zip(coo.row, coo.col):
            adjacency_list[i].append(j)
        return dict(adjacency_list)
    else:
        values = defaultdict(list)
        for i, j, v in zip(coo.row, coo.col, coo.data):
            adjacency_list[i].append(j)
            values[i].append(v)
        return dict(adjacency_list), dict(values)
