import itertools
import math
import copy

import pytest
import numpy as np
from ase import Atoms, geometry
from ase.build import molecule, bulk
from dscribe.descriptors import MBTR

from conftest import (
    assert_n_features,
    assert_dtype,
    assert_basis,
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_derivatives,
    assert_systems,
    assert_normalization,
    assert_mbtr_location,
    assert_mbtr_location_exception,
    assert_mbtr_peak,
    water,
)

# =============================================================================
# Utilities
default_k1 = {
    "k1": {
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 50},
        "weighting": {"function": "unity"},
    }
}

default_k2 = {
    "k2": {
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1 / 0.7, "sigma": 0.1, "n": 50},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
    }
}

default_k3 = {
    "k3": {
        "geometry": {"function": "angle"},
        "grid": {"min": 0, "max": 180, "sigma": 2, "n": 50},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
    }
}


def mbtr(**kwargs):
    """Returns a function that can be used to create a valid MBTR
    descriptor for a dataset.
    """

    def func(systems=None):
        species = set()
        for system in systems or []:
            species.update(system.get_atomic_numbers())
        final_kwargs = {"species": species}
        final_kwargs.update(kwargs)
        return MBTR(**final_kwargs)

    return func


def mbtr_default_k2(**kwargs):
    return mbtr(**default_k2, **kwargs)


def k2_dict(geometry_function, weighting_function):
    d = {}
    if geometry_function == "inverse_distance":
        d["geometry"] = {"function": "inverse_distance"}
        d["grid"] = {"min": 0, "max": 1.0, "sigma": 0.02, "n": 100}
    else:
        d["geometry"] = {"function": "distance"}
        d["grid"] = {"min": 0, "max": 10.0, "sigma": 0.5, "n": 100}

    if weighting_function == "exp":
        d["weighting"] = {"function": "exp", "r_cut": 9.0, "threshold": 1e-3}

    return d


def k3_dict(weighting_function):
    d = {}
    d["geometry"] = {"function": "cosine"}
    d["grid"] = {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 100}

    if weighting_function == "exp":
        d["weighting"] = {"function": "exp", "scale": 1.0, "threshold": 1e-3}

    return d


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
@pytest.mark.parametrize(
    "setup, n_features",
    [
        pytest.param(default_k1, 2 * default_k1["k1"]["grid"]["n"], id="K1"),
        pytest.param(
            default_k2, 2 * default_k2["k2"]["grid"]["n"] * 1 / 2 * (2 + 1), id="K2"
        ),
        pytest.param(
            default_k3,
            2 * default_k3["k3"]["grid"]["n"] * 1 / 2 * (2 + 1) * 2,
            id="K3",
        ),
    ],
)
def test_number_of_features(setup, n_features):
    assert_n_features(mbtr(**setup), n_features)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_dtype(dtype, sparse):
    assert_dtype(mbtr_default_k2, dtype, sparse)


@pytest.mark.parametrize(
    "n_jobs, flatten, sparse",
    [
        (1, True, False),  # Serial job, flattened, dense
        (2, True, False),  # Parallel job, flattened, dense
        (1, True, True),  # Serial job, flattened, sparse
        (2, True, True),  # Parallel job, flattened, sparse
    ],
)
def test_parallellization(n_jobs, flatten, sparse):
    assert_parallellization(mbtr_default_k2, n_jobs, flatten, sparse)


def test_no_system_modification():
    assert_no_system_modification(mbtr_default_k2)


@pytest.mark.parametrize(
    "pbc, cell",
    [
        pytest.param(
            [True, True, True], [5, 5, 5], id="Fully periodic system with cell"
        ),
        pytest.param([False, False, False], None, id="Unperiodic system with no cell"),
        pytest.param(
            [False, False, False], [0, 0, 0], id="Unperiodic system with no cell"
        ),
        pytest.param(
            [True, False, False], [5, 5, 5], id="Partially periodic system with cell"
        ),
        pytest.param(
            [True, True, True],
            [[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]],
            id="Fully periodic system with non-cubic cell",
        ),
    ],
)
def test_systems(pbc, cell):
    assert_systems(mbtr_default_k2(periodic=True), pbc, cell)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(default_k1, id="K1"),
        pytest.param(default_k2, id="K2"),
        pytest.param(default_k3, id="K3"),
    ],
)
def test_basis(setup):
    assert_basis(mbtr(**setup, periodic=True))


def test_sparse():
    assert_sparse(mbtr_default_k2)


def test_symmetries():
    assert_symmetries(mbtr_default_k2(), True, True, True)


@pytest.mark.parametrize(
    "method, periodic, normalization, k2, k3",
    [
        (
            "numerical",
            True,
            "none",
            k2_dict("inverse_distance", "exp"),
            k3_dict("exp"),
        ),
        (
            "numerical",
            False,
            "none",
            k2_dict("inverse_distance", "exp"),
            k3_dict("exp"),
        ),
        (
            "analytical",
            True,
            "none",
            k2_dict("inverse_distance", "exp"),
            k3_dict("exp"),
        ),
        (
            "analytical",
            False,
            "none",
            k2_dict("inverse_distance", "none"),
            k3_dict("none"),
        ),
        (
            "analytical",
            True,
            "none",
            k2_dict("distance", "exp"),
            k3_dict("exp"),
        ),
        (
            "analytical",
            True,
            "n_atoms",
            k2_dict("inverse_distance", "exp"),
            k3_dict("exp"),
        ),
    ],
)
def test_derivatives(method, periodic, normalization, k2, k3):
    mbtr_func = mbtr(normalization=normalization, periodic=periodic, k2=k2, k3=k3)
    assert_derivatives(mbtr_func, method, periodic, water())


@pytest.mark.parametrize("normalization", ['l2', 'n_atoms'])
def test_normalization(normalization):
    """Tests that the normalization works correctly.
    """
    assert_normalization(mbtr_default_k2, normalization)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_location(k):
    assert_mbtr_location(mbtr, k)


@pytest.mark.parametrize(
    "location",
    [
        pytest.param(["G"], id="invalid species"),
        pytest.param(["O"], id="species not specified"),
        pytest.param(["H", "H"], id="invalid k"),
    ],
)
def test_location_exceptions(location):
    assert_mbtr_location_exception(mbtr(**default_k3, species=["H"])(), location)


water_periodic = water()
water_periodic.set_pbc(True)


@pytest.mark.parametrize(
    "system,k,geometry,grid,weighting,periodic,peaks,prominence",
    [
        pytest.param(
            water(),
            1,
            {"function": "atomic_number"},
            {"min": 0, "max": 9, "sigma": 0.5, "n": 1000},
            None,
            False,
            [(("H"), [1], [2]), (("O"), [8], [1])],
            0.5,
            id="k1 finite",
        ),
        pytest.param(
            water_periodic,
            1,
            {"function": "atomic_number"},
            {"min": 0, "max": 9, "sigma": 0.5, "n": 1000},
            None,
            True,
            [(("H"), [1], [2]), (("O"), [8], [1])],
            0.5,
            id="k1 periodic",
        ),
        pytest.param(
            water(),
            2,
            {"function": "distance"},
            {"min": -1, "max": 3, "sigma": 0.5, "n": 1000},
            {"function": "unity"},
            False,
            [(("H", "H"), [1.4972204318527715], [1]), (("H", "O"), [0.95], [2])],
            0.5,
            id="k2 finite",
        ),
        pytest.param(
            Atoms(
                cell=[
                    [10, 0, 0],
                    [10, 10, 0],
                    [10, 0, 10],
                ],
                symbols=["H", "C"],
                scaled_positions=[
                    [0.1, 0.5, 0.5],
                    [0.9, 0.5, 0.5],
                ],
                pbc=True,
            ),
            2,
            {"function": "distance"},
            {"min": 0, "max": 10, "sigma": 0.5, "n": 1000},
            {"function": "exp", "scale": 0.8, "threshold": 1e-3},
            True,
            [(("H", "C"), [2, 8], np.exp(-0.8 * np.array([2, 8])))],
            0.001,
            id="k2 periodic",
        ),
        pytest.param(
            water(),
            3,
            {"function": "angle"},
            {"min": -10, "max": 180, "sigma": 5, "n": 2000},
            {"function": "unity"},
            False,
            [(("H", "H", "O"), [38], [2]), (("H", "O", "H"), [104], [1])],
            0.5,
            id="k3 finite",
        ),
        pytest.param(
            Atoms(
                cell=[
                    [10, 0, 0],
                    [0, 10, 0],
                    [0, 0, 10],
                ],
                symbols=3 * ["H"],
                scaled_positions=[
                    [0.05, 0.40, 0.5],
                    [0.05, 0.60, 0.5],
                    [0.95, 0.5, 0.5],
                ],
                pbc=True,
            ),
            3,
            {"function": "angle"},
            {"min": 0, "max": 180, "sigma": 5, "n": 2000},
            {"function": "exp", "scale": 0.85, "threshold": 1e-3},
            True,
            [
                (
                    ("H", "H", "H"),
                    [45, 90],
                    [
                        2 * np.exp(-0.85 * (2 + 2 * np.sqrt(2))),
                        np.exp(-0.85 * (2 + 2 * np.sqrt(2))),
                    ],
                )
            ],
            0.01,
            id="k3 periodic cubic 1",
        ),
        pytest.param(
            Atoms(
                cell=[[2.2, 0.0, 0.0], [0.0, 2.2, 0.0], [0.0, 0.0, 2.2]],
                positions=[
                    [0, 0, 0],
                ],
                symbols=["H"],
                pbc=True,
            ),
            3,
            {"function": "cosine"},
            {"min": -1.1, "max": 1.1, "sigma": 0.010, "n": 2000},
            {"function": "exp", "scale": 1, "threshold": 1e-4},
            True,
            [
                (
                    ("H", "H", "H"),
                    np.cos(
                        np.array(
                            [
                                180,
                                90,
                                np.arctan(np.sqrt(2)) * 180 / np.pi,
                                45,
                                np.arctan(np.sqrt(2) / 2) * 180 / np.pi,
                                0,
                            ]
                        )
                        * np.pi
                        / 180
                    ),
                    [
                        0.00044947,
                        0.00911117,
                        0.00261005,
                        0.01304592,
                        0.00261256,
                        0.00089893,
                    ],
                )
            ],
            0.0001,
            id="k3 periodic cubic 2",
        ),
        pytest.param(
            Atoms(
                cell=geometry.cellpar_to_cell([3 * 2.2, 2.2, 2.2, 30, 90, 90]),
                positions=[
                    [0, 0, 0],
                ],
                symbols=["H"],
                pbc=True,
            ),
            3,
            {"function": "cosine"},
            {"min": -1.1, "max": 1.1, "sigma": 0.01, "n": 2000},
            {"function": "exp", "scale": 1.5, "threshold": 1e-4},
            True,
            [
                (
                    ("H", "H", "H"),
                    np.cos(np.array([180, 105, 75, 51.2, 30, 23.8, 0]) * np.pi / 180),
                    [
                        0.00107715,
                        0.00044707,
                        0.00098481,
                        0.00044723,
                        0.00049224,
                        0.00044734,
                        0.00215429,
                    ],
                )
            ],
            0.00001,
            id="k3 periodic non-cubic",
        ),
    ],
)
def test_peaks(system, k, geometry, grid, weighting, periodic, peaks, prominence):
    """Tests the correct peak locations and intensities are found.
    Args:
        system: The system to test
        geometry: geometry config
        grid: grid config
        weighting: weighting config
        periodic: Whether to enable periodicity
        peaks: List of assumed peak locations and intensities
        prominence: How prominent peaks should be considered
    """
    assert_mbtr_peak(mbtr, system, k, grid, geometry, weighting, periodic, peaks, prominence)


# =============================================================================
# Tests that are specific to this descriptor.
def test_exceptions():
    """Tests different invalid parameters that should raise an
    exception.
    """
    # Weighting needs to be provided for periodic system and terms k>1
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            k2={
                **default_k2["k2"],
                "weighting": None,
            },
            periodic=True,
        )
    msg = "Periodic systems need to have a weighting function."
    assert msg == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            k2={
                **default_k2["k2"],
                "weighting": {"function": "unity"},
            },
            periodic=True,
        )
    assert msg == str(excinfo.value)

    # Invalid weighting function
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k1={
                **default_k1["k1"],
                "weighting": {"function": "exp", "threshold": 1, "scale": 1},
            },
            periodic=True,
        )
    msg = "Unknown weighting function specified for k=1. Please use one of the following: ['unity']"
    assert msg == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k2={
                **default_k2["k2"],
                "weighting": {"function": "none"},
            },
            periodic=True,
        )
    msg = "Unknown weighting function specified for k=2. Please use one of the following: ['exp', 'inverse_square', 'unity']"
    assert msg == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k3={
                **default_k3["k3"],
                "weighting": {"function": "none"},
            },
            periodic=True,
        )
    msg = "Unknown weighting function specified for k=3. Please use one of the following: ['exp', 'smooth_cutoff', 'unity']"
    assert msg == str(excinfo.value)

    # Invalid geometry function
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k1={
                "geometry": {"function": "none"},
                "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1},
            },
            periodic=False,
        )

    msg = "Unknown geometry function specified for k=1. Please use one of the following: ['atomic_number']"
    assert msg == str(excinfo.value)

    # Missing threshold
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        del setup["k2"]["weighting"]["threshold"]
        MBTR(**setup, species=[1], periodic=True)
    msg = "Missing value for 'threshold' in the k=2 weighting."
    assert msg == str(excinfo.value)

    # Missing scale or r_cut
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        del setup["k2"]["weighting"]["scale"]
        MBTR(**setup, species=[1], periodic=True)
    msg = "Provide either 'scale' or 'r_cut' in the k=2 weighting."
    assert msg == str(excinfo.value)

    # Both scale and r_cut provided
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        setup["k2"]["weighting"]["scale"] = 1
        setup["k2"]["weighting"]["r_cut"] = 1
        MBTR(**setup, species=[1], periodic=True)
    msg = "Provide either 'scale' or 'r_cut', not both in the k=2 weighting."
    assert msg == str(excinfo.value)

    # Unknown normalization
    with pytest.raises(ValueError) as excinfo:
        MBTR(**default_k2, species=[1], normalization="l2_test", periodic=True)
    msg = "Unknown normalization option given. Please use one of the following: l2, l2_each, n_atoms, none, valle_oganov."
    assert msg == str(excinfo.value)


@pytest.mark.parametrize(
    "normalize_gaussians",
    [
        pytest.param(True, id="normalized"),
        pytest.param(False, id="unnormalized"),
    ],
)
def test_gaussian_distribution(normalize_gaussians):
    """Check that the broadening follows gaussian distribution."""
    # Check with normalization
    std = 1
    start = -3
    stop = 11
    n = 500
    desc = MBTR(
        species=["H", "O"],
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": start, "max": stop, "sigma": std, "n": n},
        },
        normalize_gaussians=normalize_gaussians,
    )
    system = water()
    y = desc.create(system)
    x = np.linspace(start, stop, n)

    # Find the location of the peaks
    h_loc = desc.get_location(["H"])
    peak1_x = np.searchsorted(x, 1)
    h_feat = y[h_loc]
    peak1_y = h_feat[peak1_x]
    o_loc = desc.get_location(["O"])
    peak2_x = np.searchsorted(x, 8)
    o_feat = y[o_loc]
    peak2_y = o_feat[peak2_x]

    # Check against the analytical value
    prefactor = 1 / np.sqrt(2 * np.pi) if normalize_gaussians else 1
    gaussian = (
        lambda x, mean: 1
        / std
        * prefactor
        * np.exp(-((x - mean) ** 2) / (2 * std**2))
    )
    assert np.allclose(peak1_y, 2 * gaussian(1, 1), rtol=0, atol=0.001)
    assert np.allclose(peak2_y, gaussian(8, 8), rtol=0, atol=0.001)

    # Check the integral
    pdf = y[h_loc]
    dx = (stop - start) / (n - 1)
    sum_cum = np.sum(0.5 * dx * (pdf[:-1] + pdf[1:]))
    exp = 1 if normalize_gaussians else 1 / (1 / math.sqrt(2 * math.pi * std**2))
    assert np.allclose(sum_cum, 2 * exp, rtol=0, atol=0.001)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(default_k2, id="K2"),
        pytest.param(default_k3, id="K3"),
    ],
)
def test_periodic_translation(setup):
    """Tests that the final spectra does not change when translating atoms
    in a periodic cell. This is not trivially true unless the weight of
    distances between periodic neighbours are not halfed. Notice that the
    values of the geometry and weight functions are not equal before
    summing them up in the final graph.
    """
    # Original system with atoms separated by a cell wall
    atoms = Atoms(
        cell=[
            [10, 0, 0],
            [10, 10, 0],
            [10, 0, 10],
        ],
        symbols=["H", "H", "H", "H"],
        scaled_positions=[
            [0.1, 0.50, 0.5],
            [0.1, 0.60, 0.5],
            [0.9, 0.50, 0.5],
            [0.9, 0.60, 0.5],
        ],
        pbc=True,
    )

    # Translated system with atoms next to each other
    atoms2 = atoms.copy()
    atoms2.translate([5, 0, 0])
    atoms2.wrap()

    desc = MBTR(species=["H", "C"], **setup, periodic=True)

    # The resulting spectra should be identical
    spectra1 = desc.create(atoms)
    spectra2 = desc.create(atoms2)
    assert np.allclose(spectra1, spectra2, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(
            {
                "k1": {
                    "geometry": {"function": "atomic_number"},
                    "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 100},
                }
            },
            id="K1",
        ),
        pytest.param(
            {
                "k2": {
                    "geometry": {"function": "inverse_distance"},
                    "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 200},
                    "weighting": {
                        "function": "exp",
                        "scale": 1,
                        "threshold": 1e-3,
                    },
                }
            },
            id="K2",
        ),
        pytest.param(
            {
                "k3": {
                    "geometry": {"function": "cosine"},
                    "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 200},
                    "weighting": {
                        "function": "exp",
                        "scale": 1,
                        "threshold": 1e-3,
                    },
                }
            },
            id="K3",
        ),
    ],
)
def test_periodic_supercell_similarity(setup):
    """Tests that the output spectrum of various supercells of the same
    crystal is identical after it is normalized.
    """
    desc = MBTR(
        species=["H"],
        periodic=True,
        **setup,
        flatten=True,
        sparse=False,
        normalization="l2_each",
    )

    # Create various supercells for the FCC structure
    a1 = bulk("H", "fcc", a=2.0)  # Primitive
    a2 = a1 * [2, 2, 2]  # Supercell
    a3 = bulk("H", "fcc", a=2.0, orthorhombic=True)  # Orthorhombic
    a4 = bulk("H", "fcc", a=2.0, cubic=True)  # Conventional cubic

    output = desc.create([a1, a2, a3, a4])

    # import matplotlib.pyplot as mpl
    # mpl.plot(output[0, :])
    # mpl.plot(output[1, :])
    # mpl.plot(output[2, :])
    # mpl.plot(output[3, :])
    # mpl.show()

    # Test for equality
    assert np.allclose(output[0, :], output[1, :], atol=1e-5, rtol=0)
    assert np.allclose(output[0, :], output[2, :], atol=1e-5, rtol=0)
    assert np.allclose(output[0, :], output[3, :], atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(
            {
                "k1": {
                    "geometry": {"function": "atomic_number"},
                    "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 21},
                }
            },
            id="K1",
        ),
        pytest.param(
            {
                "k2": {
                    "geometry": {"function": "inverse_distance"},
                    "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 21},
                    "weighting": {"function": "exp", "scale": 1, "threshold": 1e-4},
                }
            },
            id="K2",
        ),
        pytest.param(
            {
                "k3": {
                    "geometry": {"function": "cosine"},
                    "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 21},
                    "weighting": {"function": "exp", "scale": 1, "threshold": 1e-4},
                }
            },
            id="K3",
        ),
    ],
)
def test_periodic_images_1(setup):
    """Tests that periodic images are handled correctly."""
    decay = 1
    desc = MBTR(
        species=[1],
        periodic=True,
        **setup,
        normalization="l2_each",
        flatten=True,
    )

    # Tests that a system has the same spectrum as the supercell of the same
    # system.
    molecule = Atoms(
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        positions=[
            [0, 0, 0],
        ],
        symbols=["H"],
    )

    a = 1.5
    molecule.set_cell([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    molecule.set_pbc(True)
    cubic_cell = desc.create(molecule)
    suce = molecule * (2, 1, 1)
    cubic_suce = desc.create(suce)

    diff = abs(np.sum(cubic_cell - cubic_suce))
    cubic_sum = abs(np.sum(cubic_cell))
    assert diff / cubic_sum < 0.05  # A 5% error is tolerated

    # Same test but for triclinic cell
    molecule.set_cell([[0.0, 2.0, 1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 0.0]])

    triclinic_cell = desc.create(molecule)
    suce = molecule * (2, 1, 1)
    triclinic_suce = desc.create(suce)

    diff = abs(np.sum(triclinic_cell - triclinic_suce))
    tricl_sum = abs(np.sum(triclinic_cell))
    assert diff / tricl_sum < 0.05

    # Testing that the same crystal, but different unit cells will have a
    # similar spectrum when they are normalized. There will be small differences
    # in the shape (due to not double counting distances)
    a1 = bulk("H", "fcc", a=2.0)
    a2 = bulk("H", "fcc", a=2.0, orthorhombic=True)
    a3 = bulk("H", "fcc", a=2.0, cubic=True)

    triclinic_cell = desc.create(a1)
    orthorhombic_cell = desc.create(a2)
    cubic_cell = desc.create(a3)

    diff1 = abs(np.sum(triclinic_cell - orthorhombic_cell))
    diff2 = abs(np.sum(triclinic_cell - cubic_cell))
    tricl_sum = abs(np.sum(triclinic_cell))
    assert diff1 / tricl_sum < 0.05
    assert diff2 / tricl_sum < 0.05
