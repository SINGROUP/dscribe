# distutils: language = c++

import numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from acsf cimport ACSF

def rebuild(rcut, g2_params, g3_params, g4_params, g5_params, atomic_numbers):
    """Function for rebuilding an ACSFWrapper from serialized arguments.
    Defined as a module-level function as member functions are not picklable.
    """
    a = ACSFWrapper()
    a.rcut = rcut
    a.g2_params = g2_params
    a.g3_params = g3_params
    a.g4_params = g4_params
    a.g5_params = g5_params
    a.atomic_numbers = atomic_numbers

    return a

cdef class ACSFWrapper:
    cdef ACSF thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self):
        self.thisptr = ACSF()

    def __reduce__(self):
        """This function is used by the pickle module when serializing this
        object.
        """
        return (rebuild, (self.rcut, self.g2_params, self.g3_params, self.g4_params, self.g5_params, self.atomic_numbers))

    def create(self, vector[vector[float]] positions, vector[int] atomic_numbers, map[vector[int], float] distances, map[int, vector[int]] neighbours, vector[int] indices):
        return np.array(self.thisptr.create(positions, atomic_numbers, distances, neighbours, indices), dtype=np.float32)

    @property
    def rcut(self):
        return self.thisptr.rCut

    @rcut.setter
    def rcut(self, value):
        self.thisptr.setRCut(value)

    @property
    def g2_params(self):
        return self.thisptr.g2Params

    @g2_params.setter
    def g2_params(self, value):
        self.thisptr.setG2Params(value)

    @property
    def g3_params(self):
        return self.thisptr.g3Params

    @g3_params.setter
    def g3_params(self, value):
        self.thisptr.setG3Params(value)

    @property
    def g4_params(self):
        return self.thisptr.g4Params

    @g4_params.setter
    def g4_params(self, value):
        self.thisptr.setG4Params(value)

    @property
    def g5_params(self):
        return self.thisptr.g5Params

    @g5_params.setter
    def g5_params(self, value):
        self.thisptr.setG5Params(value)

    @property
    def atomic_numbers(self):
        return self.thisptr.atomicNumbers

    @atomic_numbers.setter
    def atomic_numbers(self, value):
        self.thisptr.setAtomicNumbers(value)

    @property
    def n_types(self):
        return self.thisptr.nTypes

    @property
    def n_type_pairs(self):
        return self.thisptr.nTypePairs

    @property
    def n_g2(self):
        return self.thisptr.nG2

    @property
    def n_g3(self):
        return self.thisptr.nG3

    @property
    def n_g4(self):
        return self.thisptr.nG4

    @property
    def n_g5(self):
        return self.thisptr.nG5
