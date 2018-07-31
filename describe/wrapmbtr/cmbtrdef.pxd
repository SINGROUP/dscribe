from libcpp.vector cimport vector

cdef extern from "../libmbtr/cmbtr.cpp":
    pass

cdef extern from "../libmbtr/cmbtr.h":
  cdef cppclass CMBTR:
        CMBTR(vector[vector[float]], vector[int], int) except +
        vector[vector[vector[float]]] getDisplacementTensor()
