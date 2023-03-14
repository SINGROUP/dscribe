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
from ase import Atoms
import ase.geometry
import numpy as np
import dscribe.utils.geometry


class System(Atoms):
    """Represents atomic systems that are used internally by the package.
    Inherits from the ase.Atoms class, but adds the possibility to cache
    various time-consuming quantities that can be shared when creating multiple
    descriptors.
    """

    def __init__(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        wyckoff_positions=None,
        equivalent_atoms=None,
    ):
        super().__init__(
            symbols,
            positions,
            numbers,
            tags,
            momenta,
            masses,
            magmoms,
            charges,
            scaled_positions,
            cell,
            pbc,
            celldisp,
            constraint,
            calculator,
            info,
        )

        self.wyckoff_positions = wyckoff_positions
        self.equivalent_atoms = equivalent_atoms
        self._cell_inverse = None
        self._displacement_tensor = None
        self._distance_matrix = None
        self._inverse_distance_matrix = None

    @staticmethod
    def from_atoms(atoms):
        """Creates a System object from ASE.Atoms object."""
        system = System(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            tags=atoms.get_tags(),
            momenta=atoms.get_momenta(),
            masses=atoms.get_masses(),
            magmoms=atoms.get_initial_magnetic_moments(),
            charges=atoms.get_initial_charges(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
            celldisp=atoms.get_celldisp(),
            constraint=atoms._get_constraints(),
            calculator=atoms.get_calculator(),
            info=atoms.info,
        )

        return system

    def get_cell_inverse(self):
        """Get the matrix inverse of the lattice matrix."""
        if self._cell_inverse is None:
            self._cell_inverse = np.linalg.inv(self.get_cell())
        return self._cell_inverse

    def to_scaled(self, positions, wrap=False):
        """Used to transform a set of positions to the basis defined by the
        cell of this system.

        Args:
            positions (numpy.ndarray): The positions to scale
            wrap (numpy.ndarray): Whether the positions should be wrapped
                inside the cell.

        Returns:
            numpy.ndarray: The scaled positions
        """
        fractional = np.linalg.solve(self.get_cell().T, positions.T).T

        if wrap:
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    # Yes, we need to do it twice.
                    fractional[:, i] %= 1.0
                    fractional[:, i] %= 1.0

        return fractional

    def to_cartesian(self, scaled_positions, wrap=False):
        """Used to transform a set of relative positions to the cartesian basis
        defined by the cell of this system.

        Args:
            positions (numpy.ndarray): The positions to scale
            wrap (numpy.ndarray): Whether the positions should be wrapped
                inside the cell.

        Returns:
            numpy.ndarray: The cartesian positions
        """
        if wrap:
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    # Yes, we need to do it twice.
                    scaled_positions[:, i] %= 1.0
                    scaled_positions[:, i] %= 1.0

        cartesian_positions = np.dot(scaled_positions, self.get_cell())
        return cartesian_positions

    def get_displacement_tensor(self):
        """A matrix where the entry A[i, j, :] is the vector
        self.cartesian_pos[j] - self.cartesian_pos[i].

        For periodic systems the distance of an atom from itself is the
        smallest displacement of an atom from one of it's periodic copies, and
        the distance of two different atoms is the distance of two closest
        copies.

        Returns:
            np.array: 3D matrix containing the pairwise distance vectors.
        """
        if self._displacement_tensor is None:
            D, D_len = ase.geometry.geometry.get_distances(
                self.get_positions(), cell=self.get_cell(), pbc=self.get_pbc()
            )

            # Figure out the smallest basis vector and set it as
            # displacement for diagonal
            if self.get_pbc().any():
                cell = self.get_cell()
                basis_lengths = np.linalg.norm(cell, axis=1)
                min_index = np.argmin(basis_lengths)
                min_basis = cell[min_index]
                diag_indices = np.diag_indices(len(D))
                D[diag_indices] = min_basis
                diag_indices = np.diag_indices(len(D_len))
                D_len[diag_indices] = basis_lengths[min_index]

            self._displacement_tensor = D
            self._distance_matrix = D_len

        return self._displacement_tensor

    def get_distance_matrix(self):
        """Calculates the distance matrix A defined as:

        .. math::

            A_{ij} = \lvert \mathbf{r}_i - \mathbf{r}_j \\rvert

        For periodic systems the distance of an atom from itself is the
        smallest displacement of an atom from one of it's periodic copies, and
        the distance of two different atoms is the distance of two closest
        copies.

        Returns:
            np.array: Symmetric 2D matrix containing the pairwise distances.
        """
        if self._distance_matrix is None:
            self.get_displacement_tensor()
        return self._distance_matrix

    def get_distance_matrix_within_radius(
        self, radius, pos=None, output_type="coo_matrix"
    ):
        """Calculates a sparse distance matrix by only considering distances
        within a certain cutoff. Uses a k-d tree to reach O(n log(N)) time
        complexity.

        Args:
            radius(float): The cutoff radius within which distances are
                calculated. Distances outside this radius are not included.
            output_type(str): Which container to use for output data. Options:
                "dok_matrix", "coo_matrix", "dict", or "ndarray". Default:
                "dok_matrix".

        Returns:
            dok_matrix | np.array | coo_matrix | dict: Symmetric sparse 2D
            matrix containing the pairwise distances.
        """
        dmat = dscribe.utils.geometry.get_adjacency_matrix(
            radius, self.get_positions(), pos2=pos, output_type=output_type
        )
        return dmat

    def get_inverse_distance_matrix(self):
        """Calculates the inverse distance matrix A defined as:

        .. math::

            A_{ij} = \\frac{1}{\lvert \mathbf{r}_i - \mathbf{r}_j \\rvert }

        For periodic systems the distance of an atom from itself is the
        smallest displacement of an atom from one of it's periodic copies, and
        the distance of two different atoms is the distance of two closest
        copies.

        Returns:
            np.array: Symmetric 2D matrix containing the pairwise inverse
            distances.
        """
        if self._inverse_distance_matrix is None:
            distance_matrix = self.get_distance_matrix()
            with np.errstate(divide="ignore"):
                inv_distance_matrix = np.reciprocal(distance_matrix)
            self._inverse_distance_matrix = inv_distance_matrix
        return self._inverse_distance_matrix

    def set_positions(self, newpositions, apply_constraint=True):
        self._reset_structure()
        super().set_positions(newpositions, apply_constraint)

    def set_scaled_positions(self, scaled):
        self._reset_structure()
        super().set_scaled_positions(scaled)

    def set_pbc(self, pbc):
        self._reset_structure()
        super().set_pbc(pbc)

    def set_cell(self, cell, scale_atoms=False):
        self._reset_structure()
        super().set_cell(cell, scale_atoms)

    def _reset_structure(self):
        """Resets the common structural information that is cached by this
        object. The caching is done in order to share common structural
        quantities that are needed by multiple descriptors.
        """
        self._cell_inverse = None
        self._displacement_tensor = None
        self._distance_matrix = None
        self._inverse_distance_matrix = None
