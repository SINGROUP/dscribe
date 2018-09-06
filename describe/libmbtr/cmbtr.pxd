from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string

cdef extern from "cmbtr.cpp":
    pass

cdef extern from "cmbtr.h":
  cdef cppclass index3d:
        int i
        int j
        int k
        bint lessthan "operator<"(const index3d t) const:
            return this.i < t.i

  cdef cppclass CMBTR:
        CMBTR(vector[vector[float]], vector[int], map[int,int], int) except +
        vector[vector[vector[float]]] getDisplacementTensor()
        vector[vector[float]] getDistanceMatrix()
        vector[vector[float]] getInverseDistanceMatrix()
        map[pair[int,int],vector[float]] getInverseDistanceMap()
        map[string,float] getAngleCosinesCython()

