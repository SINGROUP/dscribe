import sys
import math
import numpy as np
import unittest

from dscribe.core import System
from dscribe.descriptors import ACSF
from dscribe.utils.species import symbols_to_numbers
# from dscribe.libutils.libutils import CellList
import dscribe.libutils as ut

from ase.lattice.cubic import SimpleCubicFactory
from ase.build import bulk
import ase.data


# class GeometryTests(unittest.TestCase):

    # def test_distances(self):
        # """Tests that the periodicity is taken into account when calculating
        # distances.
        # """
        # scaled_positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        # system = System(
            # scaled_positions=scaled_positions,
            # symbols=["H", "H"],
            # cell=[
                # [5, 5, 0],
                # [0, -5, -5],
                # [5, 0, 5]
            # ],
        # )
        # disp = system.get_displacement_tensor()

        # # For a non-periodic system, periodicity should not be taken into
        # # account even if cell is defined.
        # pos = system.get_positions()
        # assumed = np.array([
            # [pos[0] - pos[0], pos[1] - pos[0]],
            # [pos[0] - pos[1], pos[1] - pos[1]],
        # ])
        # # print(disp)
        # self.assertTrue(np.allclose(assumed, disp))

        # # For a periodic system, the nearest copy should be considered when
        # # comparing distances to neighbors or to self
        # system.set_pbc([True, True, True])
        # disp = system.get_displacement_tensor()
        # # print(disp)
        # assumed = np.array([
            # [[5.0, 5.0, 0.0], [-5, 0, 0]],
            # [[5, 0, 0], [5.0, 5.0, 0.0]]])
        # self.assertTrue(np.allclose(assumed, disp))

        # # Tests that the displacement tensor is found correctly even for highly
        # # non-orthorhombic systems.
        # positions = np.array([
            # [1.56909, 2.71871, 6.45326],
            # [3.9248, 4.07536, 6.45326]
        # ])
        # cell = np.array([
            # [4.7077, -2.718, 0.],
            # [0., 8.15225, 0.],
            # [0., 0., 50.]
        # ])
        # system = System(
            # positions=positions,
            # symbols=["H", "H"],
            # cell=cell,
            # pbc=True,
        # )

        # # Fully periodic with minimum image convention
        # dist_mat = system.get_distance_matrix()
        # distance = dist_mat[0, 1]

        # # The minimum image should be within the same cell
        # expected = np.linalg.norm(positions[0, :] - positions[1, :])
        # self.assertTrue(np.allclose(distance, expected))

    # def test_transformations(self):
        # """Test that coordinates are correctly transformed from scaled to
        # cartesian and back again.
        # """
        # system = System(
            # scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            # symbols=["H", "H"],
            # cell=[
                # [5, 5, 0],
                # [0, -5, -5],
                # [5, 0, 5]
            # ],
        # )

        # orig = np.array([[2, 1.45, -4.8]])
        # scal = system.to_scaled(orig)
        # cart = system.to_cartesian(scal)
        # self.assertTrue(np.allclose(orig, cart))


class DistanceTests(unittest.TestCase):

    def test_cell_list(self):
        """Tests that the cell list implementation returns identical results
        with the naive calculation
        """
        system = bulk("NaCl", spacegroup="rocksalt", a=5.64)
        pos = system.get_positions()
        num = system.get_atomic_numbers()
        cutoff = 3
        # celllist = CellList(pos, num, cutoff)


# class GaussianTests(unittest.TestCase):

    # def test_cdf(self):
        # """Test that the implementation of the gaussian value through the
        # cumulative distribution function works as expected.
        # """
        # # from scipy.stats import norm
        # from scipy.special import erf
        # # import matplotlib.pyplot as mpl

        # start = -5
        # stop = 5
        # n_points = 9
        # centers = np.array([0])
        # sigma = 1

        # area_cum = []
        # area_pdf = []

        # # Calculate errors for dfferent number of points
        # for n_points in range(2, 10):

            # axis = np.linspace(start, stop, n_points)

            # # Calculate with cumulative function (cdf)
            # dx = (stop - start)/(n_points-1)
            # x = np.linspace(start-dx/2, stop+dx/2, n_points+1)
            # pos = x[np.newaxis, :] - centers[:, np.newaxis]
            # y = 1/2*(1 + erf(pos/(sigma*np.sqrt(2))))
            # f = np.sum(y, axis=0)
            # f_rolled = np.roll(f, -1)
            # pdf_cum = (f_rolled - f)[0:-1]/dx

            # # Calculate with probability density function (pdf)
            # dist2 = axis[np.newaxis, :] - centers[:, np.newaxis]
            # dist2 *= dist2
            # f = np.sum(np.exp(-dist2/(2*sigma**2)), axis=0)
            # f *= 1/math.sqrt(2*sigma**2*math.pi)
            # pdf_pdf = f

            # # Calculate differences
            # sum_cum = np.sum(0.5*dx*(pdf_cum[:-1]+pdf_cum[1:]))
            # sum_pdf = np.sum(0.5*dx*(pdf_pdf[:-1]+pdf_pdf[1:]))
            # area_cum.append(sum_cum)
            # area_pdf.append(sum_pdf)

            # # For plotting a comparison of tre function and the PDF and CDF
            # # estimates
            # # true_axis = np.linspace(start, stop, 200)
            # # pdf_true = norm.pdf(true_axis, centers[0], sigma)  # + norm.pdf(true_axis, centers[1], sigma)
            # # mpl.plot(axis, pdf_pdf, linestyle=":", linewidth=3, color="r")
            # # mpl.plot(axis, pdf_cum, linewidth=1, color="g")
            # # mpl.plot(true_axis, pdf_true, linestyle="--", color="b")
            # # mpl.show()

        # # Check that the sum of CDF should be very close 1
        # for i in range(len(area_cum)):
            # i_cum = area_cum[i]

            # # With i == 0, the probability is always 0.5, because of only two
            # # points
            # if i > 0:
                # cum_dist = abs(i_cum-1)
                # self.assertTrue(cum_dist < 1e-2)

        # # For plotting how the differences from unity behave as a function of
        # # points used
        # # mpl.plot(area_cum, linestyle=":", linewidth=3, color="r")
        # # mpl.plot(area_pdf, linewidth=1, color="g")
        # # mpl.plot(true_axis, pdf_true, linestyle="--", color="b")
        # # mpl.show()


# class ASETests(unittest.TestCase):

    # def test_atoms_to_system(self):
        # """Tests that an ASE Atoms is succesfully converted to a System object.
        # """
        # class NaClFactory(SimpleCubicFactory):
            # "A factory for creating NaCl (B1, Rocksalt) lattices."

            # bravais_basis = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5],
                            # [0.5, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0],
                            # [0.5, 0.5, 0.5]]
            # element_basis = (0, 1, 1, 0, 1, 0, 0, 1)

        # nacl = NaClFactory()(symbol=["Na", "Cl"], latticeconstant=5.6402)
        # system = System.from_atoms(nacl)

        # self.assertTrue(np.array_equal(nacl.get_positions(), system.get_positions()))
        # self.assertTrue(np.array_equal(nacl.get_initial_charges(), system.get_initial_charges()))
        # self.assertTrue(np.array_equal(nacl.get_atomic_numbers(), system.get_atomic_numbers()))
        # self.assertTrue(np.array_equal(nacl.get_chemical_symbols(), system.get_chemical_symbols()))
        # self.assertTrue(np.array_equal(nacl.get_cell(), system.get_cell()))
        # self.assertTrue(np.array_equal(nacl.get_pbc(), system.get_pbc()))
        # self.assertTrue(np.array_equal(nacl.get_scaled_positions(), system.get_scaled_positions()))


# class SpeciesTests(unittest.TestCase):

    # def test_species(self):
        # """Tests that the species are correctly determined.
        # """
        # # As atomic number in contructor
        # d = ACSF(rcut=6.0, species=[5, 1])
        # self.assertEqual(d.species, [5, 1])          # Saves the original variable
        # self.assertTrue(np.array_equal(d._atomic_numbers, [1, 5]))  # Ordered here

        # # Set through property
        # d.species = [10, 2]
        # self.assertEqual(d.species, [10, 2])
        # self.assertTrue(np.array_equal(d._atomic_numbers, [2, 10]))  # Ordered here

        # # As chemical symbol in the contructor
        # d = ACSF(rcut=6.0, species=["O", "H"])
        # self.assertEqual(d.species, ["O", "H"])      # Saves the original variable
        # self.assertTrue(np.array_equal(d._atomic_numbers, [1, 8]))  # Ordered here

        # # Set through property
        # d.species = ["N", "Pb"]
        # self.assertEqual(d.species, ["N", "Pb"])
        # self.assertTrue(np.array_equal(d._atomic_numbers, [7, 82]))

    # def test_symbols_to_atomic_number(self):
        # """Tests that chemical symbols are correctly transformed into atomic
        # numbers.
        # """
        # for chemical_symbol in ase.data.chemical_symbols[1:]:
            # atomic_number = symbols_to_numbers([chemical_symbol])
            # true_atomic_number = ase.data.chemical_symbols.index(chemical_symbol)
            # self.assertEqual(atomic_number, true_atomic_number)

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ASETests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(GeometryTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(GaussianTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SpeciesTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
