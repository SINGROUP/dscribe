from ase import Atoms
import numpy as np


class System(Atoms):

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
            equivalent_atoms=None):

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
            info)

        self.wyckoff_positions = wyckoff_positions
        self.equivalent_atoms = equivalent_atoms
        self._cell_inverse = None
        self._displacement_tensor = None
        self._distance_matrix = None
        self._inverse_distance_matrix = None

    @staticmethod
    def from_atoms(atoms):
        """Creates a System object from ASE.Atoms object.
        """
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
            info=atoms.info)

        return system

    def get_cell_inverse(self):
        """Get the matrix inverse of the lattice matrix.
        """
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
        fractional = np.linalg.solve(
            self.get_cell(complete=True).T,
            positions.T).T

        if wrap:
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    # Yes, we need to do it twice.
                    # See the scaled_positions.py test.
                    fractional[:, i] %= 1.0
                    fractional[:, i] %= 1.0

        return fractional

    def to_cartesian(self, scaled_positions, wrap=False):
        """Used to transofrm a set of relative positions to the cartesian basis
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
                    # See the scaled_positions.py test.
                    scaled_positions[:, i] %= 1.0
                    scaled_positions[:, i] %= 1.0

        cartesian_positions = scaled_positions.dot(self.get_cell().T)
        return cartesian_positions

    def get_displacement_tensor(self):
        """A matrix where the entry A[i, j, :] is the vector
        self.cartesian_pos[i] - self.cartesian_pos[j]

        Returns:
            np.array: 3D matrix containing the pairwise distance vectors.
        """
        if self._displacement_tensor is None:

            if np.any(self.get_pbc()):
                pos = self.get_scaled_positions()
                disp_tensor = pos[:, None, :] - pos[None, :, :]

                # Take periodicity into account by wrapping coordinate elements
                # that are bigger than 0.5 or smaller than -0.5
                indices = np.where(disp_tensor > 0.5)
                disp_tensor[indices] = 1 - disp_tensor[indices]
                indices = np.where(disp_tensor < -0.5)
                disp_tensor[indices] = disp_tensor[indices] + 1
            else:
                pos = self.get_positions()
                disp_tensor = pos[:, None, :] - pos[None, :, :]

            self._displacement_tensor = disp_tensor

        return self._displacement_tensor

    def get_distance_matrix(self):
        """Calculates the distance matrix A defined as:

            A_ij = |r_i - r_j|

        Returns:
            np.array: Symmetric 2D matrix containing the pairwise distances.
        """
        if self._distance_matrix is None:
            displacement_tensor = self.get_displacement_tensor()
            distance_matrix = np.linalg.norm(displacement_tensor, axis=2)
            self._distance_matrix = distance_matrix
        return self._distance_matrix

    def get_inverse_distance_matrix(self):
        """Calculates the inverse distance matrix A defined as:

            A_ij = 1/|r_i - r_j|

        Returns:
            np.array: Symmetric 2D matrix containing the pairwise inverse
            distances.
        """
        if self._inverse_distance_matrix is None:
            distance_matrix = self.get_distance_matrix()
            with np.errstate(divide='ignore'):
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
