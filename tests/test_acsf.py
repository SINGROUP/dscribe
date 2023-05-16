import math
import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from conftest import (
    assert_n_features,
    assert_dtype,
    assert_cell,
    assert_no_system_modification,
    assert_sparse,
    assert_basis,
    assert_parallellization,
    assert_symmetries,
    assert_periodic,
    assert_derivatives,
    assert_derivatives_include,
    assert_derivatives_exclude,
    get_simple_finite,
)
from dscribe.descriptors import ACSF


# =============================================================================
# Utilities
default_g2 = [[1, 2], [4, 5]]
default_g3 = [1, 2, 3, 4]
default_g4 = [[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]]


def cutoff(R, r_cut):
    return 0.5 * (np.cos(np.pi * R / r_cut) + 1)


def acsf(**kwargs):
    """Returns a function that can be used to create a valid ACSF
    descriptor for a dataset.
    """

    def func(systems=None):
        species = set()
        for system in systems:
            species.update(system.get_atomic_numbers())
        final_kwargs = {
            "species": species,
            "g2_params": default_g2,
            "g3_params": default_g3,
            "g4_params": default_g4,
            "r_cut": 6,
        }
        final_kwargs.update(kwargs)
        return ACSF(**final_kwargs)

    return func


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
@pytest.mark.parametrize(
    "g2, g3, g4, n_features",
    [
        (None, None, None, 2),
        ([[1, 2], [4, 5]], None, None, 2 * (2 + 1)),
        (None, [1, 2, 3, 4], None, 2 * (4 + 1)),
        (None, None, [[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]], 2 + 4 * 3),
        (
            [[1, 2], [4, 5]],
            [1, 2, 3, 4],
            [[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
            2 * (1 + 2 + 4) + 4 * 3,
        ),
    ],
)
def test_number_of_features(g2, g3, g4, n_features):
    assert_n_features(acsf(g2_params=g2, g3_params=g3, g4_params=g4), n_features)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_dtype(dtype, sparse):
    assert_dtype(acsf, dtype, sparse)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("centers", ["all", "indices_fixed", "indices_variable"])
def test_parallellization(n_jobs, sparse, centers):
    assert_parallellization(acsf, n_jobs, sparse, centers)


@pytest.mark.parametrize("cell", ["collapsed_periodic", "collapsed_finite"])
def test_cell(cell):
    assert_cell(acsf, cell)


def test_no_system_modification():
    assert_no_system_modification(acsf)


def test_sparse():
    assert_sparse(acsf)


def test_symmetries():
    # Local descriptors are not permutation symmetric.
    assert_symmetries(acsf(), True, True, False)


def test_periodic():
    assert_periodic(acsf)


def test_basis():
    assert_basis(acsf(periodic=True))


@pytest.mark.parametrize("pbc", (False,))
@pytest.mark.parametrize("attach", (True,))
def test_derivatives_numerical(pbc, attach):
    assert_derivatives(acsf(periodic=pbc), "numerical", pbc, attach=attach)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_include(method):
    assert_derivatives_include(acsf(), method, True)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_exclude(method):
    assert_derivatives_exclude(acsf(), method, True)


# =============================================================================
# Tests that are specific to this descriptor.
def test_exceptions():
    """Tests different invalid parameters that should raise an
    exception.
    """
    # Invalid species
    with pytest.raises(ValueError):
        ACSF(r_cut=6.0, species=None)

    # Invalid bond_params
    with pytest.raises(ValueError):
        ACSF(r_cut=6.0, species=[1, 6, 8], g2_params=[1, 2, 3])

    # Invalid bond_cos_params
    with pytest.raises(ValueError):
        ACSF(r_cut=6.0, species=[1, 6, 8], g3_params=[[1, 2], [3, 1]])

    # Invalid bond_cos_params
    with pytest.raises(ValueError):
        ACSF(r_cut=6.0, species=[1, 6, 8], g3_params=[[1, 2, 4], [3, 1]])

    # Invalid ang4_params
    with pytest.raises(ValueError):
        ACSF(r_cut=6.0, species=[1, 6, 8], g4_params=[[1, 2], [3, 1]])

    # Invalid ang5_params
    with pytest.raises(ValueError):
        ACSF(r_cut=6.0, species=[1, 6, 8], g5_params=[[1, 2], [3, 1]])


def test_features():
    """Tests that the correct features are present in the descriptor."""
    system = get_simple_finite()
    rs = math.sqrt(2)
    kappa = math.sqrt(3)
    eta = math.sqrt(5)
    lmbd = 1
    zeta = math.sqrt(7)

    # Test against assumed values
    dist_oh = system.get_distance(0, 1)
    dist_hh = system.get_distance(0, 2)
    ang_hoh = system.get_angle(0, 1, 2) * np.pi / 180.0
    ang_hho = system.get_angle(1, 0, 2) * np.pi / 180.0
    ang_ohh = -system.get_angle(2, 0, 1) * np.pi / 180.0
    rc = 6.0

    # G1
    desc = ACSF(r_cut=rc, species=[1, 8])
    acsfg1 = desc.create(system)
    g1_ho = cutoff(dist_oh, rc)
    g1_hh = cutoff(dist_hh, rc)
    g1_oh = 2 * cutoff(dist_oh, rc)
    assert acsfg1[0, 0] == pytest.approx(g1_hh)
    assert acsfg1[0, 1] == pytest.approx(g1_ho)
    assert acsfg1[1, 0] == pytest.approx(g1_oh)

    # G2
    desc = ACSF(r_cut=6.0, species=[1, 8], g2_params=[[eta, rs]])
    acsfg2 = desc.create(system)
    g2_hh = np.exp(-eta * np.power((dist_hh - rs), 2)) * g1_hh
    g2_ho = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_ho
    g2_oh = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_oh
    assert acsfg2[0, 1] == pytest.approx(g2_hh)
    assert acsfg2[0, 3] == pytest.approx(g2_ho)
    assert acsfg2[1, 1] == pytest.approx(g2_oh)

    # G3
    desc = ACSF(r_cut=6.0, species=[1, 8], g3_params=[kappa])
    acsfg3 = desc.create(system)
    g3_hh = np.cos(dist_hh * kappa) * g1_hh
    g3_ho = np.cos(dist_oh * kappa) * g1_ho
    g3_oh = np.cos(dist_oh * kappa) * g1_oh
    assert acsfg3[0, 1] == pytest.approx(g3_hh)
    assert acsfg3[0, 3] == pytest.approx(g3_ho)
    assert acsfg3[1, 1] == pytest.approx(g3_oh)

    # G4
    desc = ACSF(r_cut=6.0, species=[1, 8], g4_params=[[eta, zeta, lmbd]])
    acsfg4 = desc.create(system)
    gauss = (
        np.exp(-eta * (2 * dist_oh * dist_oh + dist_hh * dist_hh))
        * g1_ho
        * g1_hh
        * g1_ho
    )
    g4_h_ho = (
        np.power(2, 1 - zeta) * np.power((1 + lmbd * np.cos(ang_hho)), zeta) * gauss
    )
    g4_h_oh = (
        np.power(2, 1 - zeta) * np.power((1 + lmbd * np.cos(ang_ohh)), zeta) * gauss
    )
    g4_o_hh = (
        np.power(2, 1 - zeta) * np.power((1 + lmbd * np.cos(ang_hoh)), zeta) * gauss
    )
    assert acsfg4[0, 3] == pytest.approx(g4_h_ho)
    assert acsfg4[2, 3] == pytest.approx(g4_h_oh)
    assert acsfg4[1, 2] == pytest.approx(g4_o_hh)

    # G5
    desc = ACSF(r_cut=6.0, species=[1, 8], g5_params=[[eta, zeta, lmbd]])
    acsfg5 = desc.create(system)
    gauss = np.exp(-eta * (dist_oh * dist_oh + dist_hh * dist_hh)) * g1_ho * g1_hh
    g5_h_ho = (
        np.power(2, 1 - zeta) * np.power((1 + lmbd * np.cos(ang_hho)), zeta) * gauss
    )
    g5_h_oh = (
        np.power(2, 1 - zeta) * np.power((1 + lmbd * np.cos(ang_ohh)), zeta) * gauss
    )
    g5_o_hh = (
        np.power(2, 1 - zeta)
        * np.power((1 + lmbd * np.cos(ang_hoh)), zeta)
        * np.exp(-eta * (2 * dist_oh * dist_oh))
        * g1_ho
        * g1_ho
    )
    assert acsfg5[0, 3] == pytest.approx(g5_h_ho)
    assert acsfg5[2, 3] == pytest.approx(g5_h_oh)
    assert acsfg5[1, 2] == pytest.approx(g5_o_hh)


def test_periodicity():
    """Test that periodic copies are correctly repeated and included in the
    output.
    """
    system = Atoms(
        symbols=["H"],
        positions=[[0, 0, 0]],
        cell=[2, 2, 2],
        pbc=True,
    )
    r_cut = 2.5

    # Non-periodic
    desc = ACSF(r_cut=r_cut, species=[1], periodic=False)
    feat = desc.create(system)
    assert feat.sum() == 0

    # Periodic cubic: 6 neighbours at distance 2 Å
    desc = ACSF(r_cut=r_cut, species=[1], periodic=True)
    feat = desc.create(system)
    assert feat.sum() != 0
    assert feat[0, 0] == pytest.approx(6 * cutoff(2, r_cut))

    # Periodic cubic: 6 neighbours at distance 2 Å
    # from ase.visualize import view
    r_cut = 3
    system_nacl = bulk("NaCl", "rocksalt", a=4)
    eta, zeta, lambd = 0.01, 0.1, 1
    desc = ACSF(
        r_cut=r_cut,
        g4_params=[(eta, zeta, lambd)],
        species=["Na", "Cl"],
        periodic=True,
    )
    feat = desc.create(system_nacl)

    # Cl-Cl: 12 triplets with 90 degree angle at 2 angstrom distance
    R_ij = 2
    R_ik = 2
    R_jk = np.sqrt(2) * 2
    theta = np.pi / 2
    g4_cl_cl = (
        2 ** (1 - zeta)
        * 12
        * (1 + lambd * np.cos(theta)) ** zeta
        * np.e ** (-eta * (R_ij**2 + R_ik**2 + R_jk**2))
        * cutoff(R_ij, r_cut)
        * cutoff(R_ik, r_cut)
        * cutoff(R_jk, r_cut)
    )
    assert np.allclose(feat[0, 4], g4_cl_cl, rtol=1e-6, atol=0)

    # Na-Cl: 24 triplets with 45 degree angle at sqrt(2)*2 angstrom distance
    R_ij = np.sqrt(2) * 2
    R_ik = 2
    R_jk = 2
    theta = np.pi / 4
    g4_na_cl = (
        2 ** (1 - zeta)
        * 24
        * (1 + lambd * np.cos(theta)) ** zeta
        * np.e ** (-eta * (R_ij**2 + R_ik**2 + R_jk**2))
        * cutoff(R_ij, r_cut)
        * cutoff(R_ik, r_cut)
        * cutoff(R_jk, r_cut)
    )
    assert np.allclose(feat[0, 3], g4_na_cl, rtol=1e-6, atol=0)

    # Periodic primitive FCC: 12 neighbours at distance sqrt(2)/2*5
    r_cut = 4
    system_fcc = bulk("H", "fcc", a=5)
    desc = ACSF(r_cut=r_cut, species=[1], periodic=True)
    feat = desc.create(system_fcc)
    assert feat.sum() != 0
    assert feat[0, 0] == pytest.approx(
        12 * 0.5 * (np.cos(np.pi * np.sqrt(2) / 2 * 5 / r_cut) + 1)
    )
