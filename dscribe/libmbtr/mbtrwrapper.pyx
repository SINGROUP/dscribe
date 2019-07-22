# distutils: language = c++

import numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from mbtr cimport MBTR

cdef class MBTRWrapper:
    cdef MBTR *thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, vector[vector[float]] positions, vector[int] atomic_numbers, map[int,int] atomic_number_to_index_map, int interaction_limit, vector[vector[int]] indices, bool is_local):
        self.thisptr = new MBTR(positions, atomic_numbers, atomic_number_to_index_map, interaction_limit, indices, is_local)

    def __dealloc__(self):
        del self.thisptr

    def get_gaussian(self, center, weight, start, dx, sigmasqrt2, n):
        gauss = self.thisptr.gaussian(center, weight, start, dx, sigmasqrt2, n)
        return gauss

    def get_k1(self, geom_func, weight_func, parameters, start, stop, sigma, n):
        """Cython cannot directly provide the keys as tuples, so we have to do
        the conversion here on the python side.
        """
        k1_map = self.thisptr.getK1(geom_func, weight_func, parameters, start, stop, sigma, n)
        k1_map = dict(k1_map)
        new_k1_map = {}

        for key, value in k1_map.items():
            new_key = tuple([int(key.decode("utf-8"))])
            new_k1_map[new_key] = value

        return new_k1_map

    def get_k2(self, distances, neighbours, geom_func, weight_func, parameters, start, stop, sigma, n):
        """Cython cannot directly provide the keys as tuples, so we have to do
        the conversion here on the python side.
        """
        k2_map = self.thisptr.getK2(distances, neighbours, geom_func, weight_func, parameters, start, stop, sigma, n)
        k2_map = dict(k2_map)
        new_k2_map = {}

        for key, value in k2_map.items():
            new_key = tuple(int(x) for x in key.decode("utf-8").split(","))
            new_k2_map[new_key] = value

        return new_k2_map

    def get_k3(self, distances, neighbours, geom_func, weight_func, parameters, start, stop, sigma, n):
        """Cython cannot directly provide the keys as tuples, so we have to do
        the conversion here on the python side.
        """
        k3_map = self.thisptr.getK3(distances, neighbours, geom_func, weight_func, parameters, start, stop, sigma, n)
        k3_map = dict(k3_map)
        new_k3_map = {}

        for key, value in k3_map.items():
            new_key = tuple(int(x) for x in key.decode("utf-8").split(","))
            new_k3_map[new_key] = value

        return new_k3_map

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
