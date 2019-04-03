from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "acsf.cpp":
    pass

cdef extern from "acsf.h":
    cdef cppclass ACSF:

        # Constructors
        ACSF() except +
        ACSF(float, vector[vector[float]], vector[float], vector[vector[float]], vector[vector[float]], vector[int]) except +

        # Methods
        vector[vector[float]] create(vector[vector[float]], vector[int], vector[vector[float]], vector[int])
        void setRCut(float rCut)
        void setG2Params(vector[vector[float]] g2_params)
        void setG3Params(vector[float] g3_params)
        void setG4Params(vector[vector[float]] g4_params)
        void setG5Params(vector[vector[float]] g5_params)
        void setAtomicNumbers(vector[int] atomic_numbers)

        # Attributes
        int nG2
        int nG3
        int nG4
        int nG5
        int nTypes
        int nTypePairs
        float rCut
        vector[int] atomicNumbers
        vector[vector[float]] g2Params
        vector[float] g3Params
        vector[vector[float]] g4Params
        vector[vector[float]] g5Params
