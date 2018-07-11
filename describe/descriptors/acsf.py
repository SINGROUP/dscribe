from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import os, glob
import numpy as np
from ctypes import cdll, Structure, c_int, c_double, POINTER, byref

from describe.descriptors.descriptor import Descriptor


_PATH_TO_ACSF_SO = os.path.dirname(os.path.abspath(__file__))
_ACSF_SOFILES = glob.glob( "".join([ _PATH_TO_ACSF_SO, "/../libacsf/libacsf.*so*"]))
_LIBACSF = cdll.LoadLibrary(_ACSF_SOFILES[0])


class ACSFObject(Structure):
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

# libacsf.acsf_init.argtypes = [POINTER(ACSFObject)]
# libacsf.acsf_reset.argtypes = [POINTER(ACSFObject)]
_LIBACSF.acsf_compute_acsfs.argtypes = [POINTER(ACSFObject)]


class ACSF(Descriptor):

    def __init__(self, n_atoms_max, types, bond_params=None, bond_cos_params=None, ang4_params=None, ang5_params=None, rcut=5.0, flatten=True):
        """
        Args:
         n_atoms_max (int): maximum number of atoms
         flatten (bool): Whether the output of create() should be flattened to a 1D array.

        """
        super().__init__(flatten)
        self._inited = False

        self._obj = ACSFObject()
        self._obj.alloc_atoms = 0
        self._obj.alloc_work = 0

        
        if n_atoms_max <= 0:
                raise ValueError("Maximum number of atoms n_atoms_max should be positive.")

        self._n_atoms_max = n_atoms_max

        self._types = None
        self.types = types

        self._Zs = None

        self._bond_params = None
        self.bond_params = bond_params

        self._bond_cos_params = None
        self.bond_cos_params = bond_cos_params

        
        msg = "ACSF: 2-body for each atom contains G1"
        if not (self._bond_params is None):
        	msg += ", G2[{0}]".format(self._bond_params.shape[0])
        if not (self._bond_cos_params is None):
        	msg += ", G3[{0}]".format(self._bond_cos_params.shape[0])
        print(msg+" per type({0})".format(self._obj.nTypes))
        

        self._ang4_params = None
        self.ang4_params = ang4_params
        
        msg = "ACSF: 3-body contains "
        if not (self._ang4_params is None):
        	msg += "G4[{0}] ".format(self._ang4_params.shape[0])
                # msg += " per symmetric type pair ({0})".format(self._obj.nSymTypes)
        # else:
        #	msg += "no ACSFs"


        self._ang5_params = None
        self.ang5_params = ang5_params
        
        if not (self._ang5_params is None):
            msg += "G5[{0}] ".format(self._ang5_params.shape[0])
            msg += "per symmetric type pair ({0})".format(self._obj.nSymTypes)
                
        if ((self._ang4_params is None) and (self._ang5_params is None)):
            msg += "no ACSFs"
                
        print(msg)

        msg = "ACSF: total number of ACSF per atom: 2-body {0}, 3-body {1}"
        print(msg.format((1 + self._obj.n_bond_params + self._obj.n_bond_cos_params) * self._obj.nTypes, ((self._obj.n_ang4_params + self._obj.n_ang5_params) * self._obj.nSymTypes)))

        self._rcut = None
        self.rcut = rcut

        self.positions = None
        self.distances = None

        self._acsfBuffer = None

        self.acsf_bond = None
        self.acsf_ang = None

        self._inited = True

    # --- TYPES ---
    @property
    def types(self):
            return self._types

    @types.setter
    def types(self, value):

        if self._inited:
                raise ValueError("Cannot change the atomic types.")

        if value is None:
                raise ValueError("Atomic types cannot be None.")

        pmatrix = np.array(value, dtype=np.int32)

        if pmatrix.ndim != 1:
                raise ValueError("Atomic types should be a vector of integers.")

        pmatrix = np.unique(pmatrix)
        pmatrix = np.sort(pmatrix)

        print("ACSF: setting types to: "+str(pmatrix))

        self._types = pmatrix
        self._obj.types = pmatrix.ctypes.data_as(POINTER(c_int))

        # set the internal indexer
        self._obj.nTypes = c_int(pmatrix.shape[0])
        self._obj.nSymTypes = c_int(int((pmatrix.shape[0]*(pmatrix.shape[0]+1))/2))
        

        for i in range(pmatrix.shape[0]):
                self._obj.typeID[ self._types[i]] = i
    # --- ----- ---

    # --- CUTOFF RADIUS ---
    @property
    def rcut(self):
        return self._rcut

    @rcut.setter
    def rcut(self, value):

        if value <= 0:
            raise ValueError("cutoff radius should be positive.")
        
        self._rcut = c_double(value)
        print("ACSF: cutoff radius set to {0}".format(self._rcut.value))
        self._obj.cutoff = c_double(value)
    # --- ------------- ---

    # --- BOND PARAMETERS ---
    @property
    def bond_params(self):
            return self._bond_params

    @bond_params.setter
    def bond_params(self, value):

        # TODO: check that the user input makes sense...
        # ...
        if self._inited:
                raise ValueError("Cannot change 2-body ACSF parameters.")

        # handle the disable case
        if value is None:
            # print("Disabling 2-body ACSFs...")
            self._obj.n_bond_params = 0
            self._bond_params = None
            return

        pmatrix = np.array(value, dtype=np.double)  # convert to array just to be safe!
        # print("Setting 2-body ACSFs...")

        if pmatrix.ndim != 2:
            raise ValueError("arghhh! bond_params should be a matrix with two columns (eta, Rs).")

        if pmatrix.shape[1] != 2:
            raise ValueError("arghhh! bond_params should be a matrix with two columns (eta, Rs).")

        # check that etas are positive
        if np.any(pmatrix[:,0]<=0) == True:
            print("WARNING: 2-body eta parameters should be positive numbers.")
        
        # store what the user gave in the private variable
        self._bond_params = pmatrix

        # get the number of parameter pairs
        self._obj.n_bond_params = c_int(pmatrix.shape[0])

        # convert the input list to ctypes
        #self._obj.bond_params = c_double(pmatrix.shape[0] * pmatrix.shape[1])

        #assign it
        self._obj.bond_params = self._bond_params.ctypes.data_as(POINTER(c_double))
    # --- --------------- ---

    # --- COS PARAMS ---
    @property
    def bond_cos_params(self):
        return self._bond_cos_params

    @bond_cos_params.setter
    def bond_cos_params(self, value):

        # TODO: check that the user input makes sense...
        # ...
        if self._inited:
                raise ValueError("Cannot change 2-body Cos-type ACSF parameters.")

        # handle the disable case
        if value is None:
            # print("Disabling 2-body COS-type ACSFs...")
            self._obj.n_bond_cos_params = 0
            self._bond_cos_params = None
            return

        pmatrix = np.array(value, dtype=np.double)  # convert to array just to be safe!
        # print("Setting 2-body COS-type ACSFs...")

        if pmatrix.ndim != 1:
                raise ValueError("arghhh! bond_cos_params should be a vector.")

        # store what the user gave in the private variable
        self._bond_cos_params = pmatrix

        # get the number of parameter pairs
        self._obj.n_bond_cos_params = c_int(pmatrix.shape[0])

        #assign it
        self._obj.bond_cos_params = self._bond_cos_params.ctypes.data_as(POINTER(c_double))
        # --- ---------- ---

    # --- ANG PARAMS ---
    @property
    def ang4_params(self):
            return self._ang_params

    @ang4_params.setter
    def ang4_params(self, value):

        # TODO: check that the user input makes sense...
        # ...
        if self._inited:
            raise ValueError("Cannot change 3-body G4 ACSF parameters.")

        # handle the disable case
        if value is None:
            # print("Disabling 3-body ACSFs...")
            self._obj.n_ang4_params = 0
            self._ang4_params = None
            return

        pmatrix = np.array(value, dtype=np.double)  # convert to array just to be safe!
        # print("Setting 3-body ACSFs...")

        if pmatrix.ndim != 2:
            raise ValueError("arghhh! ang4_params should be a matrix with three columns (eta, zeta, lambda).")
        if pmatrix.shape[1] != 3:
            raise ValueError("arghhh! ang4_params should be a matrix with three columns (eta, zeta, lambda).")

        # check that etas are positive
        if np.any(pmatrix[:,2]<=0) == True:
            print("WARNING: 3-body G4 eta parameters should be positive numbers.")
            
        
        # store what the user gave in the private variable
        self._ang4_params = pmatrix

        # get the number of parameter pairs
        self._obj.n_ang4_params = c_int(pmatrix.shape[0])

        # assign it
        self._obj.ang4_params = self._ang4_params.ctypes.data_as(POINTER(c_double))

    @property
    def ang5_params(self):
            return self._ang_params

    @ang5_params.setter
    def ang5_params(self, value):

        # TODO: check that the user input makes sense...
        # ...
        if self._inited:
            raise ValueError("Cannot change 3-body G5 ACSF parameters.")

        # handle the disable case
        if value is None:
            # print("Disabling 3-body ACSFs...")
            self._obj.n_ang5_params = 0
            self._ang5_params = None
            return

        pmatrix = np.array(value, dtype=np.double)  # convert to array just to be safe!
        # print("Setting 3-body ACSFs...")

        if pmatrix.ndim != 2:
            raise ValueError("arghhh! ang4_params should be a matrix with three columns (eta, zeta, lambda).")
        if pmatrix.shape[1] != 3:
            raise ValueError("arghhh! ang4_params should be a matrix with three columns (eta, zeta, lambda).")

        # check that etas are positive
        if np.any(pmatrix[:,2]<=0) == True:
            print("WARNING: 3-body G5 eta parameters should be positive numbers.")
            
        
        # store what the user gave in the private variable
        self._ang5_params = pmatrix

        # get the number of parameter pairs
        self._obj.n_ang5_params = c_int(pmatrix.shape[0])

        # assign it
        self._obj.ang5_params = self._ang5_params.ctypes.data_as(POINTER(c_double))
    # --- ---------- ---



    def describe(self, system):
        """Creates the descriptor for the given systems.

        Args:
        system (System): The system for which to create the
        descriptor.

        Returns:
        A descriptor for the system in some numerical form.
        """

        if self._types is None:
            # GIVE AN ERROR
            raise ValueError("No atomic types declared for the descriptor.")

        # copy the atomic numbers
        self._Zs = np.array(system.get_atomic_numbers(), dtype=np.int32)

        if self._Zs.shape[0] > self._n_atoms_max:
            raise ValueError("The system has more atoms than n_atoms_max.")

        self._obj.Z = self._Zs.ctypes.data_as(POINTER(c_int))
        self._obj.natm = c_int(self._Zs.shape[0])

        # check the types in the system
        typs = np.array(system.get_atomic_numbers())
        typs = np.unique(typs)
        typs = np.sort(typs)

        # check if there are types not declared in self.types
        isin = np.in1d(typs, self._types)
        isin = np.unique(isin)

        if isin.shape[0] > 1 or isin[0] is False:
            raise ValueError("The system has types that were not declared.")

        self.positions = np.array(system.get_positions(), dtype=np.double)
        self._obj.positions = self.positions.ctypes.data_as(POINTER(c_double))

        self.distances = system.get_distance_matrix()
        self._obj.distances = self.distances.ctypes.data_as(POINTER(c_double))

        # amount of ACSFs for one atom for each type pair or triplet
        self._obj.nG2 = c_int(1 + self._obj.n_bond_params + self._obj.n_bond_cos_params)
        self._obj.nG3 = c_int(self._obj.n_ang4_params + self._obj.n_ang5_params)

        ''' 
        print("ng2: ",self._obj.nG2)
        print("ng2tot: ",self._obj.nG2 * self._obj.nTypes)
        print("ng3: ",self._obj.nG3)
        print("ng3tot: ",self._obj.nG3 * self._obj.nSymTypes)
        print("buffer row len: ",self._obj.nG2 * self._obj.nTypes + self._obj.nG3 * self._obj.nSymTypes)
        print("self._obj.n_ang_params: ",self._obj.n_ang_params)
        '''
        self._acsfBuffer = np.zeros((self._n_atoms_max, self._obj.nG2 * self._obj.nTypes + self._obj.nG3 * self._obj.nSymTypes))
        self._obj.acsfs = self._acsfBuffer.ctypes.data_as(POINTER(c_double))

        _LIBACSF.acsf_compute_acsfs(byref(self._obj))

        if self.flatten is True:
            return self._acsfBuffer.flatten()

        return self._acsfBuffer

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
        int: Number of features for this descriptor.
        """

        descsize = (1 + self._obj.n_bond_params + self._obj.n_bond_cos_params) * self._obj.nTypes
        descsize += (self._obj.n_ang4_params + self._obj.n_ang5_params) * self._obj.nSymTypes

        descsize *= self._n_atoms_max

        return descsize
