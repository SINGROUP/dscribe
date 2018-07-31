# distutils: language = c++

import numpy as np
from libcpp.vector cimport vector
from cmbtrdef cimport CMBTR

cdef class PyCMBTR:
    cdef CMBTR *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, vector[vector[float]] positions, vector[int] atomic_numbers, int cell_limit):
        self.thisptr = new CMBTR(positions, atomic_numbers, cell_limit)
    def __dealloc__(self):
        del self.thisptr
    def get_displacement_tensor(self):
        return np.array(self.thisptr.getDisplacementTensor(), dtype=np.float32)
