# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

import scipy.sparse
from scipy.spatial.distance import cdist

from ase import Atoms


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


def get_extended_system(system, radial_cutoff, centers=None, return_cell_indices=False):
    """Used to create a periodically extended system. If centers are not
    specified, simply takes returns the original system multiplied by an
    integer amount of times in each direction to cover the radial cutoff. If
    centers are provided, returns the exact atoms that are within the given
    radial cutoff from the given centers.

    Args:
        original_system (ase.Atoms): The original periodic system to duplicate.
        radial_cutoff (float): The radial cutoff to use in constructing the
            extended system.
        centers (np.ndarray): Array of xyz-coordinates from which the distance
            is calculated. If provided, these centers are used to calculate the
            exact distance and only atoms within the radial cutoff from these
            centers are returned.
        return_cell_indices (boolean): Whether to return an array of cell
            indices for each atom in the extended system.

    Returns:
        ase.Atoms | tuple: If return_cell_indices is False, returns the new
        extended system. Else returns a tuple containing the new extended
        system as the first entry and the index of the periodically repeated
        cell for each atom as the second entry.
    """
    # Determine the upper limit of how many copies we need in each cell vector
    # direction. We take as many copies as needed to reach the radial cutoff.
    # Notice that we need to use vectors that are perpendicular to the cell
    # vectors to ensure that the correct atoms are included for non-cubic cells.
    cell = np.array(system.get_cell())
    a1, a2, a3 = cell[0], cell[1], cell[2]
    b1 = np.cross(a2, a3, axis=0)
    b2 = np.cross(a3, a1, axis=0)
    b3 = np.cross(a1, a2, axis=0)
    p1 = np.dot(a1, b1) / np.dot(b1, b1) * b1  # Projections onto perpendicular vectors
    p2 = np.dot(a2, b2) / np.dot(b2, b2) * b2
    p3 = np.dot(a3, b3) / np.dot(b3, b3) * b3
    xyz_arr = np.linalg.norm(np.array([p1, p2, p3]), axis=1)
    cell_images = np.ceil(radial_cutoff/xyz_arr)
    nx = int(cell_images[0])
    ny = int(cell_images[1])
    nz = int(cell_images[2])
    n_copies_axis = np.array([nx, ny, nz], dtype=int)

    # If no centers are given, and the cell indices are not requested, simply
    # return the multiplied system. This is much faster.
    if centers is None and not return_cell_indices:

        n_atoms = len(system)
        n_rep = np.product(2*n_copies_axis+1)  # Number of repeated copies
        ext_pos = np.tile(system.get_positions(), (n_rep, 1))

        # Calculate the extended system positions so that the original cell
        # stays in place: both in space and in index
        i_curr = 0
        for m0 in np.append(np.arange(0, nx+1), np.arange(-nx, 0)):
            for m1 in np.append(np.arange(0, ny+1), np.arange(-ny, 0)):
                for m2 in np.append(np.arange(0, nz+1), np.arange(-nz, 0)):
                    ext_pos[i_curr:i_curr+n_atoms] += np.dot((m0, m1, m2), cell)
                    i_curr += n_atoms

        ext_symbols = np.tile(system.get_atomic_numbers(), n_rep)
        extended_system = Atoms(
            positions=ext_pos,
            symbols=ext_symbols,
        )

        return extended_system

    # If centers are given and/or cell indices are needed, the process is done
    # one cell at a time to keep track of the cell inded and the distances.
    # This is a bit slower.
    else:
        # We need to specify that the relative positions should not be wrapped.
        # Otherwise the repeated systems may overlap with the positions taken
        # with get_positions()
        relative_pos = np.array(system.get_scaled_positions(wrap=False))
        numbers = np.array(system.numbers)
        cartesian_pos = np.array(system.get_positions())

        # Create copies of the cell but keep track of the atoms in the
        # original cell
        num_extended = []
        pos_extended = []
        num_extended.append(numbers)
        pos_extended.append(cartesian_pos)
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = np.array([0, 0, 1])
        cell_indices = [np.zeros((len(system), 3), dtype=int)]

        for i in range(-n_copies_axis[0], n_copies_axis[0]+1):
            for j in range(-n_copies_axis[1], n_copies_axis[1]+1):
                for k in range(-n_copies_axis[2], n_copies_axis[2]+1):
                    if i == 0 and j == 0 and k == 0:
                        continue

                    # Calculate the positions of the copied atoms and filter
                    # out the atoms that are farther away than the given
                    # cutoff.
                    num_copy = np.array(numbers)
                    pos_copy = np.array(relative_pos)

                    pos_shifted = pos_copy-i*a-j*b-k*c
                    pos_copy_cartesian = np.dot(pos_shifted, cell)

                    # Only distances to the atoms within the interaction limit
                    # are considered.
                    distances = cdist(pos_copy_cartesian, centers)
                    weight_mask = distances < radial_cutoff

                    # Create a boolean mask that says if the atom is within the
                    # range from at least one atom in the original cell
                    valids_mask = np.any(weight_mask, axis=1)

                    if np.any(valids_mask):
                        valid_pos = pos_copy_cartesian[valids_mask]
                        valid_num = num_copy[valids_mask]
                        valid_ind = np.tile(np.array([i, j, k], dtype=int), (len(valid_num), 1))

                        pos_extended.append(valid_pos)
                        num_extended.append(valid_num)
                        cell_indices.append(valid_ind)

        pos_extended = np.concatenate(pos_extended)
        num_extended = np.concatenate(num_extended)
        cell_indices = np.vstack(cell_indices)

        extended_system = Atoms(
            positions=pos_extended,
            numbers=num_extended,
            cell=cell,
            pbc=False
        )

        return extended_system, cell_indices


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Args:
        arrays(sequence of arrays): The arrays from which the product is
            created.
        out(ndarray): Array to place the cartesian product in.

    Returns:
        ndarray: 2-D array of shape (M, len(arrays)) containing cartesian
        products formed of input arrays.

    Example:
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out
