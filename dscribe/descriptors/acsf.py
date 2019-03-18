# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import os
import glob

import numpy as np

from scipy.sparse import coo_matrix

from ctypes import cdll, Structure, c_int, c_double, POINTER, byref

from dscribe.descriptors.descriptor import Descriptor
from dscribe.utils.species import get_atomic_numbers


_PATH_TO_ACSF_SO = os.path.dirname(os.path.abspath(__file__))
_ACSF_SOFILES = glob.glob( "".join([ _PATH_TO_ACSF_SO, "/../libacsf/libacsf.*so*"]))
_LIBACSF = cdll.LoadLibrary(_ACSF_SOFILES[0])


class ACSFObject(Structure):
    """Wrapper class for the ACSF C library.
    """
    _fields_ = [
        ('natm', c_int),
        ('Z', POINTER(c_int)),
        ('positions', POINTER(c_double)),
        ('nTypes', c_int),
        ('types', POINTER(c_int)),
        ('typeID', (c_int*100)),
        ('nSymTypes', c_int),
        ('cutoff', c_double),

        ('n_bond_params', c_int),
        ('bond_params', POINTER(c_double)),

        ('n_bond_cos_params', c_int),
        ('bond_cos_params', POINTER(c_double)),

        ('n_ang4_params', c_int),
        ('ang4_params', POINTER(c_double)),

        ('n_ang5_params', c_int),
        ('ang5_params', POINTER(c_double)),

        ('distances', POINTER(c_double)),
        ('nG2', c_int),
        ('nG3', c_int),

        ('acsfs', POINTER(c_double))
    ]

_LIBACSF.acsf_compute_acsfs.argtypes = [POINTER(ACSFObject)]
_LIBACSF.acsf_compute_acsfs_some.argtypes = [POINTER(ACSFObject), POINTER(c_int), c_int]


class ACSF(Descriptor):
    """Implementation of Atom-Centered Symmetry Functions. Currently valid for
    finite systems only.

    Notice that the species of the central atom is not encoded in the output,
    only the surrounding environment is encoded. In a typical application one
    can train a different model for each central species.

    For reference, see:
        "Atom-centered symmetry functions for constructing high-dimensional
        neural network potentials", Jörg Behler, The Journal of Chemical
        Physics, 134, 074106 (2011), https://doi.org/10.1063/1.3553717
    """
    def __init__(
        self,
        rcut,
        g2_params=None,
        g3_params=None,
        g4_params=None,
        g5_params=None,
        species=None,
        atomic_numbers=None,
        sparse=False
    ):
        """
        Args:
            rcut (float): The smooth cutoff value. This cutoff value is
                used throughout the calculations for all symmetry functions.
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical speices as low as possible is
                preferable.
            atomic_numbers (iterable): A list of the atomic numbers that should
                be taken into account in the descriptor. Deprecated in favour of
                the species-parameters, but provided for
                backwards-compatibility.
            g2_params (n*2 np.ndarray): A list of pairs of :math:`\eta` and
                :math:`R_s` parameters for :math:`G^2` functions.
            g3_params (n*1 np.ndarray): A list of :math:`\kappa` parameters for
                :math:`G^3` functions.
            g4_params (n*3 np.ndarray): A list of triplets of :math:`\eta`,
                :math:`\zeta` and  :math:`\lambda` parameters for :math:`G^4` functions.
            g5_params (n*3 np.ndarray): A list of triplets of :math:`\eta`,
                :math:`\zeta` and  :math:`\lambda` parameters for :math:`G^5` functions.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(flatten=True, sparse=sparse)

        self._obj = ACSFObject()
        self._obj.alloc_atoms = 0
        self._obj.alloc_work = 0

        self._Zs = None

        # Setup the involved chemical species
        species = self.get_species_definition(species, atomic_numbers)
        self.species = species

        self.set_g2_params(g2_params)
        self.set_g3_params(g3_params)
        self.set_g4_params(g4_params)
        self.set_g5_params(g5_params)
        self.set_rcut(rcut)

        self.positions = None
        self.distances = None

        self._acsfBuffer = None

        self.acsf_bond = None
        self.acsf_ang = None

    def create(self, system, positions=None):
        """Creates the descriptor for the given systems.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            positions (iterable): Indices of the atoms around which the ACSF
                will be returned. If no positions defined, ACSF will be created
                for all atoms in the system.

        Returns:
            np.ndarray | scipy.sparse.coo_matrix: The ACSF output for the
            given system and positions. The return type depends on the
            'sparse'-attribute. The first dimension is given by the number of
            positions and the second dimension is determined by the
            get_number_of_features()-function.
        """
        # Transform the input system into the internal System-object
        system = self.get_system(system)

        # Make sure that periodicity is not taken into account
        system.set_pbc(False)

        # Copy the atomic numbers
        self._Zs = np.array(system.get_atomic_numbers(), dtype=np.int32)
        self._obj.Z = self._Zs.ctypes.data_as(POINTER(c_int))

        # Set the number of atoms
        n_atoms = len(system)
        self._obj.natm = c_int(n_atoms)

        # Check if there are types that have not been declared
        self.check_atomic_numbers(system.get_atomic_numbers())

        # Setup pointer to the atomic positions
        self.positions = np.array(system.get_positions(), dtype=np.double)
        self._obj.positions = self.positions.ctypes.data_as(POINTER(c_double))

        # Setup pointer to the distance matrix
        self.distances = system.get_distance_matrix()
        self._obj.distances = self.distances.ctypes.data_as(POINTER(c_double))

        # Amount of ACSFs for one atom for each type pair or triplet
        self._obj.nG2 = c_int(1 + self._obj.n_bond_params + self._obj.n_bond_cos_params)
        self._obj.nG3 = c_int(self._obj.n_ang4_params + self._obj.n_ang5_params)

        # We allocate memory for the ACSF of all atoms in the system, because
        # the current ACSF library expects this full-size memory buffer.
        self._acsfBuffer = np.zeros((n_atoms, self._obj.nG2 * self._obj.nTypes + self._obj.nG3 * self._obj.nSymTypes))
        self._obj.acsfs = self._acsfBuffer.ctypes.data_as(POINTER(c_double))

        # Create C-compatible list of atomic indices for which the ACSF is calculated
        if positions is None:
            indices = np.arange(len(system), dtype=np.int32)
        else:
            indices = np.array(positions, dtype=np.int32)
        indices_ptr = indices.ctypes.data_as(POINTER(c_int))

        # Calculate ACSF for the given atomic indices
        _LIBACSF.acsf_compute_acsfs_some(byref(self._obj), indices_ptr, c_int(len(indices)))

        # Retrieve only the relevant part of the output, rest are zeros.
        final_output = self._acsfBuffer[indices]

        # Return sparse matrix if requested
        if self._sparse:
            final_output = coo_matrix(final_output)

        return final_output

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        descsize = (1 + self._obj.n_bond_params + self._obj.n_bond_cos_params) * self._obj.nTypes
        descsize += (self._obj.n_ang4_params + self._obj.n_ang5_params) * self._obj.nSymTypes

        return int(descsize)

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        """Used to check the validity of given atomic numbers and to initialize
        the C-memory layout for them.

        Args:
            value(iterable): Chemical species either as a list of atomic
                numbers or list of chemical symbols.
        """
        # The species are stored as atomic numbers for internal use.
        self._set_species(value)

        # Here we setup the C memory layout
        pmatrix = np.array(self._atomic_numbers, dtype=np.int32)
        pmatrix = np.unique(pmatrix)
        pmatrix = np.sort(pmatrix)
        self._obj.types = pmatrix.ctypes.data_as(POINTER(c_int))
        self._obj.nTypes = c_int(pmatrix.shape[0])  # Set the internal indexer
        self._obj.nSymTypes = c_int(int((pmatrix.shape[0]*(pmatrix.shape[0]+1))/2))

        for i in range(pmatrix.shape[0]):
            self._obj.typeID[ pmatrix[i]] = i

    def set_rcut(self, value):
        """Used to check the validity of given radial cutoff and to initialize
        the C-memory layout for it.

        Args:
            value(float): Radial cutoff.
        """
        if value <= 0:
            raise ValueError("Vutoff radius should be positive.")

        self._obj.cutoff = c_double(value)

    def set_g2_params(self, value):
        """Used to check the validity of given G2 parameters and to
        initialize the C-memory layout for them.

        Args:
            value(n*3 array): List of G2 parameters.
        """
        # Handle the disable case
        if value is None:
            self._obj.n_bond_params = 0
            self._bond_params = None
            return

        # Convert to array just to be safe!
        pmatrix = np.array(value, dtype=np.double)

        if pmatrix.ndim != 2:
            raise ValueError("bond_params should be a matrix with two columns (eta, Rs).")

        if pmatrix.shape[1] != 2:
            raise ValueError("bond_params should be a matrix with two columns (eta, Rs).")

        # Check that etas are positive
        if np.any(pmatrix[:, 0] <= 0) is True:
            raise ValueError("2-body eta parameters should be positive numbers.")

        # Store what the user gave in the private variable
        self._bond_params = pmatrix

        # Get the number of parameter pairs
        self._obj.n_bond_params = c_int(pmatrix.shape[0])

        # Assign it
        self._obj.bond_params = self._bond_params.ctypes.data_as(POINTER(c_double))

    def set_g3_params(self, value):
        """Used to check the validity of given G3 parameters and to
        initialize the C-memory layout for them.

        Args:
            value(array): List of G3 parameters.
        """
        # Handle the disable case
        if value is None:
            self._obj.n_bond_cos_params = 0
            self._bond_cos_params = None
            return

        # Convert to array just to be safe!
        pmatrix = np.array(value, dtype=np.double)

        if pmatrix.ndim != 1:
                raise ValueError("arghhh! bond_cos_params should be a vector.")

        # Store what the user gave in the private variable
        self._bond_cos_params = pmatrix

        # Get the number of parameter pairs
        self._obj.n_bond_cos_params = c_int(pmatrix.shape[0])

        # Assign it
        self._obj.bond_cos_params = self._bond_cos_params.ctypes.data_as(POINTER(c_double))

    def set_g4_params(self, value):
        """Used to check the validity of given G4 parameters and to
        initialize the C-memory layout for them.

        Args:
            value(n*3 array): List of G4 parameters.
        """
        # Handle the disable case
        if value is None:
            self._obj.n_ang4_params = 0
            self._ang4_params = None
            return

        # Convert to array just to be safe!
        pmatrix = np.array(value, dtype=np.double)

        if pmatrix.ndim != 2:
            raise ValueError("arghhh! ang4_params should be a matrix with three columns (eta, zeta, lambda).")
        if pmatrix.shape[1] != 3:
            raise ValueError("arghhh! ang4_params should be a matrix with three columns (eta, zeta, lambda).")

        # Check that etas are positive
        if np.any(pmatrix[:, 2] <= 0) is True:
            raise ValueError("3-body G4 eta parameters should be positive numbers.")

        # Store what the user gave in the private variable
        self._ang4_params = pmatrix

        # Get the number of parameter pairs
        self._obj.n_ang4_params = c_int(pmatrix.shape[0])

        # Assign it
        self._obj.ang4_params = self._ang4_params.ctypes.data_as(POINTER(c_double))

    def set_g5_params(self, value):
        """Used to check the validity of given G5 parameters and to
        initialize the C-memory layout for them.

        Args:
            value(n*3 array): List of G5 parameters.
        """
        # handle the disable case
        if value is None:
            self._obj.n_ang5_params = 0
            self._ang5_params = None
            return

        # Convert to array just to be safe!
        pmatrix = np.array(value, dtype=np.double)

        if pmatrix.ndim != 2:
            raise ValueError("ang4_params should be a matrix with three columns (eta, zeta, lambda).")
        if pmatrix.shape[1] != 3:
            raise ValueError("ang4_params should be a matrix with three columns (eta, zeta, lambda).")

        # Check that etas are positive
        if np.any(pmatrix[:, 2] <= 0) is True:
            raise ValueError("3-body G5 eta parameters should be positive numbers.")

        # Store what the user gave in the private variable
        self._ang5_params = pmatrix

        # Get the number of parameter pairs
        self._obj.n_ang5_params = c_int(pmatrix.shape[0])

        # Assign it
        self._obj.ang5_params = self._ang5_params.ctypes.data_as(POINTER(c_double))
