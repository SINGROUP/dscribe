from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string

cdef extern from "mbtr.cpp":
    pass

cdef extern from "mbtr.h":
  cdef cppclass MBTR:
        MBTR(vector[vector[float]], vector[int], map[int,int], int, vector[vector[int]], bool) except +
        map[string,vector[float]] getK1(string, string, map[string, float], float, float, float, float) except +
        map[string,vector[float]] getK2(vector[vector[float]], vector[vector[int]], string, string, map[string, float], float, float, float, float) except +
        map[string,vector[float]] getK3(vector[vector[float]], vector[vector[int]], string, string, map[string, float], float, float, float, float) except +
        vector[map[string,vector[float]]] getK2Local(vector[vector[float]], vector[vector[int]], vector[int], string, string, map[string, float], float, float, float, float) except +
