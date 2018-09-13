# distutils: language = c++

import numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from cmbtr cimport CMBTR

cdef class CMBTRWrapper:
    cdef CMBTR *thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, vector[vector[float]] positions, vector[int] atomic_numbers, map[int,int] atomicNumberToIndexMap, int cell_limit):
        self.thisptr = new CMBTR(positions, atomic_numbers, atomicNumberToIndexMap, cell_limit)

    def __dealloc__(self):
        del self.thisptr

    def get_displacement_tensor(self):
        return np.array(self.thisptr.getDisplacementTensor(), dtype=np.float32)

    def get_distance_matrix(self):
        return np.array(self.thisptr.getDistanceMatrix(), dtype=np.float32)

    # def get_inverse_distance_matrix(self):
        # return np.array(self.thisptr.getInverseDistanceMatrix(), dtype=np.float32)

    # def get_inverse_distance_map(self):
        # return self.thisptr.getInverseDistanceMap()

    def get_k2_map(self, geom_func, weight_func, parameters):
        # Get the angle map and convert the keys to tuples. Cython cannot
        # directly provide the keys as tuples, so we have to do the conversion
        # here on the python side.
        geom_map, dist_map = self.thisptr.getK2MapCython(geom_func, weight_func, parameters);
        new_geom_map = {}
        new_dist_map = {}

        for key, value in geom_map.items():
            new_key = tuple(int(x) for x in key.decode("utf-8").split(","))
            new_geom_map[new_key] = value
            new_dist_map[new_key] = dist_map[key]

        return new_geom_map, new_dist_map
    def get_k3_map(self, geom_func, weight_func, parameters):
        # Get the angle map and convert the keys to tuples. Cython cannot
        # directly provide the keys as tuples, so we have to do the conversion
        # here on the python side.
        geom_map, dist_map = self.thisptr.getK3MapCython(geom_func, weight_func, parameters);
        new_geom_map = {}
        new_dist_map = {}

        for key, value in geom_map.items():
            new_key = tuple(int(x) for x in key.decode("utf-8").split(","))
            new_geom_map[new_key] = value
            new_dist_map[new_key] = dist_map[key]

        return new_geom_map, new_dist_map
