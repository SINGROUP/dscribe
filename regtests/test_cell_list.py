import math
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import dscribe.ext


@pytest.mark.parametrize(
    "system,cutoff",
    [
        pytest.param(
            bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True) * [3, 3, 3],
            0.5 * 3 * 5.64,
            id="cubic, cutoff < cell",
        ),
        pytest.param(
            bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True) * [3, 3, 3],
            1 * 3 * 5.64,
            id="cubic, cutoff == cell",
        ),
        pytest.param(
            bulk("NaCl", crystalstructure="rocksalt", a=5.64, cubic=True) * [3, 3, 3],
            1.5 * 3 * 5.64,
            id="cubic, cutoff > cell",
        ),
        pytest.param(
            Atoms(
                cell=[[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
                positions=[
                    [0, 0, 0],
                    [0.95, 0, 0],
                    [
                        0.95 * (1 + math.cos(76 / 180 * math.pi)),
                        0.95 * math.sin(76 / 180 * math.pi),
                        0.0,
                    ],
                ],
                symbols=["H", "O", "H"],
            )
            * [
                3,
                3,
                3,
            ],
            2,
            id="triclinic, cutoff < cell",
        ),
    ],
)
def test_cell_list_index(system, cutoff):
    """Tests that the cell list implementation returns identical results
    with the naive calculation.
    """
    pos = system.get_positions()
    all_distances_naive = system.get_all_distances()
    cell_list = dscribe.ext.CellList(pos, cutoff)
    for idx in range(len(system)):
        # Get indices and distances to neighbours using cell list
        result = cell_list.get_neighbours_for_index(idx)
        indices = list(result.keys())
        distances = list([x[0] for x in result.values()])
        sort_order = np.argsort(indices)
        indices = np.array(indices)[sort_order]
        distances = np.array(distances)[sort_order]

        # Calculate the reference indices and distances of atoms with a simple
        # O(n^2) method
        indices_naive = np.where(np.linalg.norm(pos - pos[idx], axis=1) <= cutoff)[0]
        indices_naive = indices_naive[indices_naive != idx]
        distances_naive = all_distances_naive[idx, indices_naive]

        # Compare
        assert np.array_equal(indices, indices_naive)
        assert np.allclose(distances, distances_naive, atol=1e-16, rtol=1e-16)


@pytest.mark.parametrize(
    "system,cutoff,positions",
    [
        pytest.param(
            Atoms(
                positions=[[0, 0, 0]],
                symbols=["H"],
            ),
            5,
            [([4, 3, 0], [(0, 5.0)]), ([4, 3, 0.001], [])],
            id="system < cutoff",
        ),
        pytest.param(
            Atoms(
                positions=[[0, 0, 0], [1, 1, 1]],
                symbols=["H", "H"],
            ),
            0.2,
            [([500, 500, 500], []), ([-500, -500, -500], [])],
            id="position outside bins",
        ),
        pytest.param(
            Atoms(
                positions=[[0, 0, 0], [1, 1, 1]],
                symbols=["H", "H"],
            ),
            0.2,
            [([-0.2, 0, 0], [(0, 0.2)]), ([1, 1.2, 1], [(1, 0.2)])],
            id="position at the edge of bins",
        ),
    ],
)
def test_cell_list_position(system, cutoff, positions):
    """Tests that the cell list implementation returns identical results
    with the naive calculation.
    """
    pos = system.get_positions()
    cell_list = dscribe.ext.CellList(pos, cutoff)
    for (x, y, z), neighbours in positions:
        result = cell_list.get_neighbours_for_position(x, y, z)
        for index, distance in neighbours:
            assert result[index][0] == pytest.approx(distance, rel=1e-8, abs=0)
        assert len(result) == len(neighbours)
