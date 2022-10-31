import numpy as np
import pytest
from ase import Atoms
from ase.visualize import view

import dscribe.ext
from dscribe.utils.geometry import get_extended_system


@pytest.mark.parametrize(
    "system,cutoff,atomic_numbers,positions,indices,cell_indices,interactive_atoms",
    [
        pytest.param(
            Atoms(
                cell=[[0.0, 1.0, 1.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
                positions=[[0, 0, 0], [1, 1, 0], [1, 0, 1]],
                symbols=["C", "O", "H"],
                pbc=True,
            ),
            1,
            [6, 8, 1] * 27,
            [
                [0, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [2, 2, 0],
                [3, 3, 0],
                [3, 2, 1],
                [-2, -2, 0],
                [-1, -1, 0],
                [-1, -2, 1],
                [2, 0, 2],
                [3, 1, 2],
                [3, 0, 3],
                [4, 2, 2],
                [5, 3, 2],
                [5, 2, 3],
                [0, -2, 2],
                [1, -1, 2],
                [1, -2, 3],
                [-2, 0, -2],
                [-1, 1, -2],
                [-1, 0, -1],
                [0, 2, -2],
                [1, 3, -2],
                [1, 2, -1],
                [-4, -2, -2],
                [-3, -1, -2],
                [-3, -2, -1],
                [0, 1, 1],
                [1, 2, 1],
                [1, 1, 2],
                [2, 3, 1],
                [3, 4, 1],
                [3, 3, 2],
                [-2, -1, 1],
                [-1, 0, 1],
                [-1, -1, 2],
                [2, 1, 3],
                [3, 2, 3],
                [3, 1, 4],
                [4, 3, 3],
                [5, 4, 3],
                [5, 3, 4],
                [0, -1, 3],
                [1, 0, 3],
                [1, -1, 4],
                [-2, 1, -1],
                [-1, 2, -1],
                [-1, 1, 0],
                [0, 3, -1],
                [1, 4, -1],
                [1, 3, 0],
                [-4, -1, -1],
                [-3, 0, -1],
                [-3, -1, 0],
                [0, -1, -1],
                [1, 0, -1],
                [1, -1, 0],
                [2, 1, -1],
                [3, 2, -1],
                [3, 1, 0],
                [-2, -3, -1],
                [-1, -2, -1],
                [-1, -3, 0],
                [2, -1, 1],
                [3, 0, 1],
                [3, -1, 2],
                [4, 1, 1],
                [5, 2, 1],
                [5, 1, 2],
                [0, -3, 1],
                [1, -2, 1],
                [1, -3, 2],
                [-2, -1, -3],
                [-1, 0, -3],
                [-1, -1, -2],
                [0, 1, -3],
                [1, 2, -3],
                [1, 1, -2],
                [-4, -3, -3],
                [-3, -2, -3],
                [-3, -3, -2],
            ],
            [0, 1, 2] * 27,
            np.repeat(np.arange(27), 3),
            {0, 1, 2},
            id="fully periodic system",
        ),
    ],
)
def test_extend_system(
    system, cutoff, atomic_numbers, positions, indices, cell_indices, interactive_atoms
):
    system_cpp = dscribe.ext.System(
        system.get_positions(),
        system.get_atomic_numbers(),
        system.get_cell(),
        system.get_pbc(),
    )
    ext_system_cpp = dscribe.ext.extend_system(system_cpp, cutoff)
    # view(system)
    # ext_system_ase = Atoms(
    #     positions=ext_system_cpp.get_positions(),
    #     numbers=ext_system_cpp.get_atomic_numbers(),
    #     cell=ext_system_cpp.get_cell(),
    #     pbc=ext_system_cpp.get_pbc(),
    # )
    # view(ext_system_ase)

    assert np.array_equal(ext_system_cpp.get_atomic_numbers(), atomic_numbers)
    assert np.allclose(ext_system_cpp.get_positions(), positions)
    assert np.array_equal(ext_system_cpp.get_indices(), indices)
    assert np.array_equal(ext_system_cpp.get_cell_indices(), cell_indices)
    assert ext_system_cpp.get_interactive_atoms() == interactive_atoms
