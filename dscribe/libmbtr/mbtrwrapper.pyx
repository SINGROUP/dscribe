# distutils: language = c++

import numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from mbtr cimport MBTR

cdef class MBTRWrapper:
    cdef MBTR *thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, vector[vector[float]] positions, vector[int] atomic_numbers, map[int,int] atomic_number_to_index_map, int interaction_limit, vector[vector[int]] indices, bool is_local):
        self.thisptr = new MBTR(positions, atomic_numbers, atomic_number_to_index_map, interaction_limit, indices, is_local)

    def __dealloc__(self):
        del self.thisptr

    # def get_displacement_tensor(self):
        # return np.array(self.thisptr.getDisplacementTensor(), dtype=np.float32)

    # def get_distance_matrix(self):
        # return np.array(self.thisptr.getDistanceMatrix(), dtype=np.float32)

    def get_k1_geoms_and_weights(self, geom_func, weight_func, parameters):
        """Cython cannot directly provide the keys as tuples, so we have to do
        the conversion here on the python side.
        """
        geom_map, weight_map = self.thisptr.getK1GeomsAndWeightsCython(geom_func, weight_func, parameters);
        new_geom_map = {}
        new_weight_map = {}

        for key, value in geom_map.items():
            new_key = tuple([int(key.decode("utf-8"))])
            new_geom_map[new_key] = value
            new_weight_map[new_key] = weight_map[key]

        return new_geom_map, new_weight_map

    def get_k2_geoms_and_weights(self, distances, neighbours,  geom_func, weight_func, parameters):
        """Cython cannot directly provide the keys as tuples, so we have to do
        the conversion here on the python side.
        """
        geom_map, weight_map = self.thisptr.getK2GeomsAndWeightsCython(distances, neighbours, geom_func, weight_func, parameters);
        new_geom_map = {}
        new_weight_map = {}

        for key, value in geom_map.items():
            new_key = tuple(int(x) for x in key.decode("utf-8").split(","))
            new_geom_map[new_key] = value
            new_weight_map[new_key] = weight_map[key]

        return new_geom_map, new_weight_map

    def get_k3_geoms_and_weights(self, distances, neighbours, geom_func, weight_func, parameters):
        """Cython cannot directly provide the keys as tuples, so we have to do
        the conversion here on the python side.
        """
        geom_map, weight_map = self.thisptr.getK3GeomsAndWeightsCython(distances, neighbours, geom_func, weight_func, parameters);
        new_geom_map = {}
        new_weight_map = {}

        for key, value in geom_map.items():
            new_key = tuple(int(x) for x in key.decode("utf-8").split(","))
            new_geom_map[new_key] = value
            new_weight_map[new_key] = weight_map[key]

        return new_geom_map, new_weight_map
