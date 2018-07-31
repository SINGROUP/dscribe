from libcpp.vector cimport vector

cdef extern from "cmbtr.cpp":
    pass

cdef extern from "cmbtr.h":
  cdef cppclass CMBTR:
        CMBTR(vector[vector[float]], vector[int], int) except +
        vector[vector[vector[double]]] getDisplacementTensor()
