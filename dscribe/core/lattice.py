# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np


class Lattice(object):
    """
    A lattice object.  Essentially a matrix with conversion matrices. In
    general, it is assumed that length units are in Angstroms and angles are in
    degrees unless otherwise stated.
    """
    def __init__(self, matrix):
        """
        Create a lattice from any sequence of 9 numbers. Note that the sequence
        is assumed to be read one row at a time. Each row represents one
        lattice vector.

        Args:
            matrix: Sequence of numbers in any form. Examples of acceptable
                input.
                i) An actual numpy array.
                ii) [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                iii) [1, 0, 0 , 0, 1, 0, 0, 0, 1]
                iv) (1, 0, 0, 0, 1, 0, 0, 0, 1)
                Each row should correspond to a lattice vector.
                E.g., [[10, 0, 0], [20, 10, 0], [0, 0, 30]] specifies a lattice
                with lattice vectors [10, 0, 0], [20, 10, 0] and [0, 0, 30].
        """
        m = np.array(matrix, dtype=np.float64).reshape((3, 3))
        lengths = np.sqrt(np.sum(m ** 2, axis=1))
        self._lengths = lengths
        self._matrix = m
        self._angles = None
        self._inv_matrix = None

    @property
    def matrix(self):
        """Copy of matrix representing the Lattice"""
        return np.copy(self._matrix)

    @property
    def inv_matrix(self):
        """
        Inverse of lattice matrix.
        """
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self._matrix)
        return self._inv_matrix

    def get_cartesian_coords(self, fractional_coords):
        """
        Returns the cartesian coordinates given fractional coordinates.

        Args:
            fractional_coords (3x1 array): Fractional coords.

        Returns:
            Cartesian coordinates
        """
        return np.dot(fractional_coords, self._matrix)

    def get_fractional_coords(self, cart_coords):
        """
        Returns the fractional coordinates given cartesian coordinates.

        Args:
            cart_coords (3x1 array): Cartesian coords.

        Returns:
            Fractional coordinates.
        """
        return np.dot(cart_coords, self.inv_matrix)

    @property
    def lengths(self):
        if self._lengths is None:
            lengths = np.linalg.norm(self._matrix, axis=1)
            self._lengths = lengths
        return self._lengths

    @property
    def angles(self):
        """
        Returns the angles (alpha, beta, gamma) of the lattice.
        """
        if self._angles is None:
            # Angles
            angles = np.zeros(3)
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                angles[i] = np.dot(
                    self._matrix[j],
                    self._matrix[k]) / (self.lengths[j] * self.lengths[k])
            angles = np.clip(angles, -1.0, 1.0)
            self._angles = np.arccos(angles) * 180. / np.pi
        return self._angles

    @property
    def abc(self):
        """
        Lengths of the lattice vectors, i.e. (a, b, c)
        """
        return tuple(self.lengths)

    @property
    def alpha(self):
        """
        Angle alpha of lattice in degrees.
        """
        return self._angles[0]

    @property
    def beta(self):
        """
        Angle beta of lattice in degrees.
        """
        return self._angles[1]

    @property
    def gamma(self):
        """
        Angle gamma of lattice in degrees.
        """
        return self._angles[2]

    @property
    def volume(self):
        """
        Volume of the unit cell.
        """
        m = self._matrix
        return abs(np.dot(np.cross(m[0], m[1]), m[2]))

    @property
    def lengths_and_angles(self):
        """
        Returns (lattice lengths, lattice angles).
        """
        return tuple(self.lengths), tuple(self.angles)

    @property
    def reciprocal_lattice(self):
        """
        Return the reciprocal lattice. Note that this is the standard
        reciprocal lattice used for solid state physics with a factor of 2 *
        pi. If you are looking for the crystallographic reciprocal lattice,
        use the reciprocal_lattice_crystallographic property.
        The property is lazily generated for efficiency.
        """
        try:
            return self._reciprocal_lattice
        except AttributeError:
            v = self.inv_matrix.T
            self._reciprocal_lattice = Lattice(v * 2 * np.pi)
            return self._reciprocal_lattice

    @property
    def reciprocal_lattice_crystallographic(self):
        """
        Returns the *crystallographic* reciprocal lattice, i.e., no factor of
        2 * pi.
        """
        return Lattice(self.reciprocal_lattice.matrix / (2 * np.pi))

    def get_points_in_sphere(self, frac_points, center, r, zip_results=True):
        """
        Find all points within a sphere from the point taking into account
        periodic boundary conditions. This includes sites in other periodic
        images.

        Algorithm:

        1. place sphere of radius r in crystal and determine minimum supercell
           (parallelpiped) which would contain a sphere of radius r. for this
           we need the projection of a_1 on a unit vector perpendicular
           to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
           determine how many a_1"s it will take to contain the sphere.

           Nxmax = r * length_of_b_1 / (2 Pi)

        2. keep points falling within r.

        Args:
            frac_points: All points in the lattice in fractional coordinates.
            center: Cartesian coordinates of center of sphere.
            r: radius of sphere.
            zip_results (bool): Whether to zip the results together to group by
                 point, or return the raw fcoord, dist, index arrays

        Returns:
            if zip_results:
                [(fcoord, dist, index) ...] since most of the time, subsequent
                processing requires the distance.
            else:
                fcoords, dists, inds
        """
        # TODO: refactor to use lll matrix (nmax will be smaller)
        recp_len = np.array(self.reciprocal_lattice.abc) / (2 * np.pi)
        nmax = float(r) * recp_len + 0.01

        pcoords = self.get_fractional_coords(center)
        center = np.array(center)

        n = len(frac_points)
        fcoords = np.array(frac_points) % 1
        indices = np.arange(n)

        mins = np.floor(pcoords - nmax)
        maxes = np.ceil(pcoords + nmax)
        arange = np.arange(start=mins[0], stop=maxes[0])
        brange = np.arange(start=mins[1], stop=maxes[1])
        crange = np.arange(start=mins[2], stop=maxes[2])
        arange = arange[:, None] * np.array([1, 0, 0])[None, :]
        brange = brange[:, None] * np.array([0, 1, 0])[None, :]
        crange = crange[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] +\
            crange[None, None, :]

        shifted_coords = fcoords[:, None, None, None, :] + \
            images[None, :, :, :, :]

        cart_coords = self.get_cartesian_coords(fcoords)
        cart_images = self.get_cartesian_coords(images)
        coords = cart_coords[:, None, None, None, :] + \
            cart_images[None, :, :, :, :]
        coords -= center[None, None, None, None, :]
        coords **= 2
        d_2 = np.sum(coords, axis=4)

        within_r = np.where(d_2 <= r ** 2)
        if zip_results:
            return list(zip(shifted_coords[within_r], np.sqrt(d_2[within_r]),
                            indices[within_r[0]]))
        else:
            return shifted_coords[within_r], np.sqrt(d_2[within_r]), \
                indices[within_r[0]]
