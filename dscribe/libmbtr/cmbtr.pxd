from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string

cdef extern from "cmbtr.cpp":
    pass

cdef extern from "cmbtr.h":
  cdef cppclass CMBTR:
        CMBTR(vector[vector[float]], vector[int], map[int,int], int, bool) except +
        vector[vector[vector[float]]] getDisplacementTensor()
        vector[vector[float]] getDistanceMatrix()
        vector[vector[float]] getInverseDistanceMatrix()
        map[pair[int,int],vector[float]] getInverseDistanceMap()
        pair[map[string,vector[float]], map[string,vector[float]]] getK1GeomsAndWeightsCython(string, string, map[string, float]) except +
        pair[map[string,vector[float]], map[string,vector[float]]] getK2GeomsAndWeightsCython(string, string, map[string, float]) except +
        pair[map[string,vector[float]], map[string,vector[float]]] getK3GeomsAndWeightsCython(string, string, map[string, float]) except +
