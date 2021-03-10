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
import math
import unittest
import itertools
from pprint import pprint as pp
import numpy as np

import sparse as sp
import scipy
import scipy.sparse
from scipy.integrate import tplquad
from scipy.linalg import sqrtm

from dscribe.descriptors import SOAP

from ase import Atoms
from ase.build import molecule
from ase.visualize import view
from ase.build import bulk

H2 = Atoms(
    cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
    positions=[
        [-0.5, 0, 0],
        [0.5, 0, 0],
    ],
    symbols=["H", "H"],
)
H = Atoms(
    positions=[[0.5, 0.5, 0.5]],
    symbols=["H"],
)
H2O = molecule("H2O")
CO2 = molecule("CO2")


class SoapDerivativeTests(unittest.TestCase):
    def test_exceptions(self):
        """Test the derivative interface."""
        soap = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            sparse=False,
        )
        positions = [[0.0, 0.0, 0.0]]

        # Test that trying to get analytical derivatives with averaged output
        # raises an exception
        soap.sparse = False
        soap.average = "inner"
        with self.assertRaises(ValueError):
            soap.derivatives(H2, positions=positions, method="analytical")
        soap.average = "off"

        # Test that trying to get analytical derivatives with polynomial basis
        # raises an exception, but the numerical ones work.
        soap_poly = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            rbf="polynomial",
            sparse=False,
        )
        with self.assertRaises(ValueError):
            soap_poly.derivatives(H2, positions=positions, method="analytical")
        soap_poly.derivatives(H2, positions=positions, method="numerical")

        # Test that trying to provide both include and exclude raises an error
        with self.assertRaises(ValueError):
            soap.derivatives(H2, include=[0], exclude=[1])

        # Test that trying to get derivatives with periodicicity on raises an
        # exception
        soap = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            rbf="gto",
            sparse=False,
            periodic=True,
        )
        with self.assertRaises(ValueError):
            soap.derivatives(H2, positions=positions, method="analytical")
        with self.assertRaises(ValueError):
            soap.derivatives(H2, positions=positions, method="numerical")

    def test_return_descriptor(self):
        soap = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            rbf="gto",
            sparse=False,
            periodic=False,
        )
        s = soap.derivatives(H2O, method="analytical", return_descriptor=False)
        D, d = soap.derivatives(H2O, method="analytical", return_descriptor=True)
        s = soap.derivatives(H2O, method="numerical", return_descriptor=False)
        D, d = soap.derivatives(H2O, method="numerical", return_descriptor=True)

    def test_include(self):
        soap = SOAP(
            species=[1, 6, 8],
            rcut=3,
            nmax=2,
            lmax=1,
            rbf="gto",
            sparse=False,
            periodic=False,
        )

        for method in ["numerical", "analytical"]:
            # Invalid include options
            with self.assertRaises(ValueError):
                soap.derivatives(H2O, include=[], method=method)
            with self.assertRaises(ValueError):
                soap.derivatives(H2O, include=[3], method=method)
            with self.assertRaises(ValueError):
                soap.derivatives(H2O, include=[-1], method=method)

            # Test that correct atoms are included and in the correct order
            D1, d1 = soap.derivatives(H2O, include=[2, 0], method=method)
            D2, d2 = soap.derivatives(H2O, method=method)
            self.assertTrue(np.array_equal(D1[:, 0], D2[:, 2]))
            self.assertTrue(np.array_equal(D1[:, 1], D2[:, 0]))

            # Test that using single list and multiple samples works
            D1, d1 = soap.derivatives([H2O, CO2], include=[1, 0], method=method)
            D2, d2 = soap.derivatives([H2O, CO2], method=method)
            self.assertTrue(np.array_equal(D1[:, :, 0], D2[:, :, 1]))
            self.assertTrue(np.array_equal(D1[:, :, 1], D2[:, :, 0]))

    def test_exclude(self):
        soap = SOAP(
            species=[1, 6, 8],
            rcut=3,
            nmax=2,
            lmax=1,
            rbf="gto",
            sparse=False,
            periodic=False,
        )

        for method in ["numerical", "analytical"]:
            # Invalid exclude options
            with self.assertRaises(ValueError):
                soap.derivatives(H2O, exclude=[3], method=method)
            with self.assertRaises(ValueError):
                soap.derivatives(H2O, exclude=[-1], method=method)

            # Test that correct atoms are excluded and in the correct order
            D1, d1 = soap.derivatives(H2O, exclude=[1], method=method)
            D2, d2 = soap.derivatives(H2O, method=method)
            self.assertTrue(np.array_equal(D1[:, 0], D2[:, 0]))
            self.assertTrue(np.array_equal(D1[:, 1], D2[:, 2]))

            # Test that using single list and multiple samples works
            D1, d1 = soap.derivatives([H2O, CO2], exclude=[1], method=method)
            D2, d2 = soap.derivatives([H2O, CO2], method=method)
            self.assertTrue(np.array_equal(D1[:, :, 0], D2[:, :, 0]))
            self.assertTrue(np.array_equal(D1[:, :, 1], D2[:, :, 2]))

    def test_parallel(self):
        """Tests parallel output validity for both dense and sparse output."""
        for sparse in [False, True]:
            desc = SOAP(
                species=[1, 6, 7, 8],
                rcut=5,
                nmax=3,
                lmax=3,
                sigma=1,
                periodic=False,
                crossover=True,
                average="off",
                sparse=sparse,
            )
            n_features = desc.get_number_of_features()

            samples = [molecule("CO"), molecule("NO"), molecule("OH")]
            centers = [[0], [0], [0]]

            # Determining number of jobs based on the amount of CPUs
            # desc.derivatives(system=samples, n_jobs=-1, only_physical_cores=False)
            # desc.derivatives(system=samples, n_jobs=-1, only_physical_cores=True)

            # Perhaps most common scenario: more systems than jobs, using all
            # centers and indices
            der, des = desc.derivatives(
                system=samples,
                n_jobs=2,
            )
            self.assertTrue(der.shape == (3, 2, 2, 3, n_features))
            self.assertTrue(des.shape == (3, 2, n_features))
            assumed_der = np.empty((3, 2, 2, 3, n_features))
            assumed_des = np.empty((3, 2, n_features))
            desc.sparse = False
            assumed_der[0, :], assumed_des[0, :] = desc.derivatives(
                samples[0], n_jobs=1
            )
            assumed_der[1, :], assumed_des[1, :] = desc.derivatives(
                samples[1], n_jobs=1
            )
            assumed_der[2, :], assumed_des[2, :] = desc.derivatives(
                samples[2], n_jobs=1
            )
            desc._sparse = sparse
            if sparse:
                der = der.todense()
                des = des.todense()
            self.assertTrue(np.allclose(assumed_der, der))
            self.assertTrue(np.allclose(assumed_des, des))

            # More systems than jobs, using all centers and indices, not requesting
            # descriptors
            der = desc.derivatives(
                system=samples,
                return_descriptor=False,
                n_jobs=2,
            )
            self.assertTrue(der.shape == (3, 2, 2, 3, n_features))
            assumed_der = np.empty((3, 2, 2, 3, n_features))
            desc._sparse = False
            assumed_der[0, :] = desc.derivatives(
                samples[0], n_jobs=1, return_descriptor=False
            )
            assumed_der[1, :] = desc.derivatives(
                samples[1], n_jobs=1, return_descriptor=False
            )
            assumed_der[2, :] = desc.derivatives(
                samples[2], n_jobs=1, return_descriptor=False
            )
            desc._sparse = sparse
            if sparse:
                der = der.todense()
            self.assertTrue(np.allclose(assumed_der, der))

            # More systems than jobs, using custom indices as centers
            der, des = desc.derivatives(
                system=samples,
                positions=centers,
                n_jobs=2,
            )
            self.assertTrue(der.shape == (3, 1, 2, 3, n_features))
            self.assertTrue(des.shape == (3, 1, n_features))
            assumed_der = np.empty((3, 1, 2, 3, n_features))
            assumed_des = np.empty((3, 1, n_features))
            desc._sparse = False
            assumed_der[0, :], assumed_des[0, :] = desc.derivatives(
                samples[0], centers[0], n_jobs=1
            )
            assumed_der[1, :], assumed_des[1, :] = desc.derivatives(
                samples[1], centers[1], n_jobs=1
            )
            assumed_der[2, :], assumed_des[2, :] = desc.derivatives(
                samples[2], centers[2], n_jobs=1
            )
            desc._sparse = sparse
            if sparse:
                der = der.todense()
                des = des.todense()
            self.assertTrue(np.allclose(assumed_der, der))
            self.assertTrue(np.allclose(assumed_des, des))

            # More systems than jobs, using custom cartesian centers
            centers = [[[0, 1, 2]], [[2, 1, 0]], [[1, 2, 0]]]
            der, des = desc.derivatives(
                system=samples,
                positions=centers,
                n_jobs=2,
            )
            self.assertTrue(der.shape == (3, 1, 2, 3, n_features))
            self.assertTrue(des.shape == (3, 1, n_features))
            assumed_der = np.empty((3, 1, 2, 3, n_features))
            assumed_des = np.empty((3, 1, n_features))
            desc._sparse = False
            assumed_der[0, :], assumed_des[0, :] = desc.derivatives(
                samples[0], centers[0], n_jobs=1
            )
            assumed_der[1, :], assumed_des[1, :] = desc.derivatives(
                samples[1], centers[1], n_jobs=1
            )
            assumed_der[2, :], assumed_des[2, :] = desc.derivatives(
                samples[2], centers[2], n_jobs=1
            )
            desc._sparse = sparse
            if sparse:
                der = der.todense()
                des = des.todense()
            self.assertTrue(np.allclose(assumed_der, der))
            self.assertTrue(np.allclose(assumed_des, des))

            # Includes
            includes = [[0], [0], [0]]
            der, des = desc.derivatives(
                system=samples,
                include=includes,
                n_jobs=2,
            )
            self.assertTrue(der.shape == (3, 2, 1, 3, n_features))
            self.assertTrue(des.shape == (3, 2, n_features))
            assumed_der = np.empty((3, 2, 1, 3, n_features))
            assumed_des = np.empty((3, 2, n_features))
            desc._sparse = False
            assumed_der[0, :], assumed_des[0, :] = desc.derivatives(
                samples[0], include=includes[0], n_jobs=1
            )
            assumed_der[1, :], assumed_des[1, :] = desc.derivatives(
                samples[1], include=includes[1], n_jobs=1
            )
            assumed_der[2, :], assumed_des[2, :] = desc.derivatives(
                samples[2], include=includes[2], n_jobs=1
            )
            desc._sparse = sparse
            if sparse:
                der = der.todense()
                des = des.todense()
            self.assertTrue(np.allclose(assumed_der, der))
            self.assertTrue(np.allclose(assumed_des, des))

            # Excludes
            excludes = [[0], [0], [0]]
            der, des = desc.derivatives(
                system=samples,
                exclude=excludes,
                n_jobs=2,
            )
            self.assertTrue(der.shape == (3, 2, 1, 3, n_features))
            self.assertTrue(des.shape == (3, 2, n_features))
            assumed_der = np.empty((3, 2, 1, 3, n_features))
            assumed_des = np.empty((3, 2, n_features))
            desc._sparse = False
            assumed_der[0, :], assumed_des[0, :] = desc.derivatives(
                samples[0], exclude=excludes[0], n_jobs=1
            )
            assumed_der[1, :], assumed_des[1, :] = desc.derivatives(
                samples[1], exclude=excludes[1], n_jobs=1
            )
            assumed_der[2, :], assumed_des[2, :] = desc.derivatives(
                samples[2], exclude=excludes[2], n_jobs=1
            )
            desc._sparse = sparse
            if sparse:
                der = der.todense()
                des = des.todense()
            self.assertTrue(np.allclose(assumed_der, der))
            self.assertTrue(np.allclose(assumed_des, des))

            # Test averaged output
            desc.average = "inner"
            positions = [[0], [0, 1], [1]]
            der, des = desc.derivatives(
                system=samples,
                positions=positions,
                n_jobs=2,
            )
            self.assertTrue(der.shape == (3, 1, 2, 3, n_features))
            self.assertTrue(des.shape == (3, 1, n_features))
            desc._sparse = False
            assumed_der = np.empty((3, 1, 2, 3, n_features))
            assumed_des = np.empty((3, 1, n_features))
            assumed_der[0, :], assumed_des[0, :] = desc.derivatives(
                samples[0], positions=positions[0], n_jobs=1
            )
            assumed_der[1, :], assumed_des[1, :] = desc.derivatives(
                samples[1], positions=positions[1], n_jobs=1
            )
            assumed_der[2, :], assumed_des[2, :] = desc.derivatives(
                samples[2], positions=positions[2], n_jobs=1
            )
            desc._sparse = sparse
            if sparse:
                der = der.todense()
                des = des.todense()
            self.assertTrue(np.allclose(assumed_der, der))
            self.assertTrue(np.allclose(assumed_des, des))

            # Variable size list output, as the systems have a different size
            desc.average = "off"
            samples = [molecule("CO"), molecule("NO2"), molecule("OH")]
            der, des = desc.derivatives(
                system=samples,
                n_jobs=2,
            )
            self.assertTrue(isinstance(der, list))
            self.assertTrue(der[0].shape == (2, 2, 3, n_features))
            self.assertTrue(der[1].shape == (3, 3, 3, n_features))
            self.assertTrue(der[2].shape == (2, 2, 3, n_features))
            desc._sparse = False
            for i in range(len(samples)):
                assumed_der, assumed_des = desc.derivatives(samples[i], n_jobs=1)
                i_der = der[i]
                i_des = des[i]
                if sparse:
                    i_der = i_der.todense()
                    i_des = i_des.todense()
                self.assertTrue(np.allclose(assumed_der, i_der))
                self.assertTrue(np.allclose(assumed_des, i_des))
            desc._sparse = sparse

    def test_numerical(self):
        """Test numerical values against a naive python implementation."""
        # Elaborate test system with multiple species, non-cubic cell, and
        # close-by atoms.
        a = 1
        system = (
            Atoms(
                symbols=["C", "H", "O"],
                cell=[[0, a, a], [a, 0, a], [a, a, 0]],
                scaled_positions=[
                    [0, 0, 0],
                    [1 / 3, 1 / 3, 1 / 3],
                    [2 / 3, 2 / 3, 2 / 3],
                ],
                pbc=[True, True, True],
            )
            * (3, 3, 3)
        )
        # view(system)

        # Two centers: one in the middle, one on the edge.
        centers = [np.sum(system.get_cell(), axis=0) / 2, [0, 0, 0]]

        h = 0.0001
        n_atoms = len(system)
        n_comp = 3

        # The maximum error depends on how big the system is. With a small
        # system the error is smaller for non-periodic systems than the
        # corresponding error when periodicity is turned on. The errors become
        # equal (~1e-5) when the size of the system is increased.
        for periodic in [False]:
            for rbf in ["gto", "polynomial"]:
                for average in ["off", "outer", "inner"]:
                    soap = SOAP(
                        species=[1, 8, 6],
                        rcut=3,
                        nmax=4,
                        lmax=4,
                        rbf=rbf,
                        sparse=False,
                        average=average,
                        crossover=True,
                        periodic=periodic,
                        dtype="float64",  # The numerical derivatives require double precision
                    )
                    n_features = soap.get_number_of_features()
                    if average != "off":
                        n_centers = 1
                        derivatives_python = np.zeros((n_atoms, n_comp, n_features))
                    else:
                        n_centers = len(centers)
                        derivatives_python = np.zeros(
                            (n_centers, n_atoms, n_comp, n_features)
                        )
                    d0 = soap.create(system, centers)
                    coeffs = [-1.0 / 2.0, 1.0 / 2.0]
                    deltas = [-1.0, 1.0]
                    for i_atom in range(len(system)):
                        for i_center in range(n_centers):
                            for i_comp in range(3):
                                for i_stencil in range(2):
                                    if average == "off":
                                        i_cent = [centers[i_center]]
                                    else:
                                        i_cent = centers
                                    system_disturbed = system.copy()
                                    i_pos = system_disturbed.get_positions()
                                    i_pos[i_atom, i_comp] += h * deltas[i_stencil]
                                    system_disturbed.set_positions(i_pos)
                                    d1 = soap.create(system_disturbed, i_cent)
                                    if average != "off":
                                        derivatives_python[i_atom, i_comp, :] += (
                                            coeffs[i_stencil] * d1 / h
                                        )
                                    else:
                                        derivatives_python[
                                            i_center, i_atom, i_comp, :
                                        ] += (coeffs[i_stencil] * d1[0, :] / h)

                    # Calculate with central finite difference implemented in C++.
                    # Try both cartesian centers and indices.
                    for c in [centers]:
                        derivatives_cpp, d_cpp = soap.derivatives(
                            system, positions=c, method="numerical"
                        )

                        # Test that descriptor values are correct
                        self.assertTrue(np.allclose(d0, d_cpp, atol=1e-6))

                        # Compare values
                        # print(np.abs(derivatives_python).max())
                        # print(derivatives_python[0,1,:,:])
                        # print(derivatives_cpp[0,0,:,:])
                        self.assertTrue(
                            np.allclose(derivatives_python, derivatives_cpp, atol=2e-5)
                        )

    def test_sparse(self):
        """Test that the sparse values are identical to the dense ones."""
        positions = H2O.get_positions()

        soap = SOAP(
            species=["H", "O"],
            rcut=3,
            nmax=1,
            lmax=1,
            sparse=False,
            crossover=False,
        )
        D_dense, d_dense = soap.derivatives(
            H2O, positions=positions, method="analytical"
        )
        soap.sparse = True
        D_sparse, d_sparse = soap.derivatives(
            H2O, positions=positions, method="analytical"
        )
        self.assertTrue(np.allclose(D_dense, D_sparse.todense()))
        self.assertTrue(np.allclose(d_dense, d_sparse.todense()))


class SoapDerivativeComparisonTests(unittest.TestCase):
    def test_no_crossover(self):
        """Tests the analytical soap derivatives implementation without
        crossover against the numerical implementation.
        """
        soap = SOAP(
            species=["H", "O"],
            rcut=3,
            nmax=9,
            lmax=9,
            sparse=False,
            crossover=False,
        )
        positions = H2O.get_positions()
        derivatives_cpp, d_num = soap.derivatives(
            H2O, positions=positions, method="numerical"
        )
        derivatives_anal, d_anal = soap.derivatives(
            H2O, positions=positions, method="analytical"
        )
        self.assertTrue(
            np.allclose(derivatives_cpp, derivatives_anal, rtol=1e-6, atol=1e-6)
        )

    def test_crossover(self):
        """Tests the analytical soap derivatives implementation with crossover
        against the numerical implementation.
        """
        soap = SOAP(
            species=["H", "O"],
            rcut=3,
            nmax=9,
            lmax=9,
            sparse=False,
            crossover=True,
        )
        positions = H2O.get_positions()
        derivatives_cpp, d_num = soap.derivatives(
            H2O, positions=positions, method="numerical"
        )
        derivatives_anal, d_anal = soap.derivatives(
            H2O, positions=positions, method="analytical"
        )
        self.assertTrue(
            np.allclose(derivatives_cpp, derivatives_anal, rtol=1e-6, atol=1e-6)
        )

    def test_descriptor_output(self):
        """Tests that the descriptor is output correctly both for numerical and
        analytical version.
        """
        soap = SOAP(
            species=["H", "O"],
            rcut=3,
            nmax=3,
            lmax=3,
            crossover=True,
        )
        positions = H2O.get_positions()
        _, d_num = soap.derivatives(H2O, positions=positions, method="numerical")
        _, d_anal = soap.derivatives(H2O, positions=positions, method="analytical")
        d = soap.create(H2O, positions=positions)
        self.assertTrue(np.allclose(d, d_num))
        self.assertTrue(np.allclose(d, d_anal))

    def test_combinations(self):
        """Tests that different combinations of centers/atoms work as intended
        and are equal between analytical/numerical code.
        """
        # Elaborate test system with multiple species, non-cubic cell, and
        # close-by atoms.
        a = 1
        system = (
            Atoms(
                symbols=["C", "C", "C"],
                cell=[[0, a, a], [a, 0, a], [a, a, 0]],
                scaled_positions=[
                    [0, 0, 0],
                    [1 / 3, 1 / 3, 1 / 3],
                    [2 / 3, 2 / 3, 2 / 3],
                ],
                pbc=[True, True, True],
            )
            * (3, 3, 3)
        )

        soap = SOAP(
            species=[6],
            # species=[1, 6, 8], #TODO: Does not pass if there are extra elements?!
            rcut=3,
            nmax=1,
            lmax=1,
            rbf="gto",
            sparse=False,
            average="off",
            crossover=True,
            periodic=False,
        )
        centers = [5, 3, 4]
        include = [3, 0, 2]

        # The typical full set: derivatives for all atoms, all atoms act as
        # centers.
        derivatives_n, d_n = soap.derivatives(system, method="numerical")
        derivatives_a, d_a = soap.derivatives(system, method="analytical")
        self.assertTrue(np.allclose(derivatives_n, derivatives_a, rtol=1e-6, atol=1e-6))
        self.assertTrue(np.allclose(d_n, d_a, rtol=1e-6, atol=1e-6))

        # Derivatives for all atoms, only some atoms act as centers
        derivatives_n, d_n = soap.derivatives(
            system, positions=centers, method="numerical"
        )
        derivatives_a, d_a = soap.derivatives(
            system, positions=centers, method="analytical"
        )
        self.assertTrue(np.allclose(derivatives_n, derivatives_a, rtol=1e-6, atol=1e-6))
        self.assertTrue(np.allclose(d_n, d_a, rtol=1e-6, atol=1e-6))

        # Derivatives for some atoms, all atoms act as centers
        derivatives_n, d_n = soap.derivatives(
            system, include=include, method="numerical"
        )
        derivatives_a, d_a = soap.derivatives(
            system, include=include, method="analytical"
        )
        self.assertTrue(np.allclose(derivatives_n, derivatives_a, rtol=1e-6, atol=1e-6))
        self.assertTrue(np.allclose(d_n, d_a, rtol=1e-6, atol=1e-6))

        # Mixed set of derivatives and centers
        derivatives_n, d_n = soap.derivatives(
            system, positions=centers, include=include, method="numerical"
        )
        derivatives_a, d_a = soap.derivatives(
            system, positions=centers, include=include, method="analytical"
        )
        self.assertTrue(np.allclose(derivatives_n, derivatives_a, rtol=1e-6, atol=1e-6))
        self.assertTrue(np.allclose(d_n, d_a, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeTests))
    suites.append(
        unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeComparisonTests)
    )
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
