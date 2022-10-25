import time
import itertools
import math
import copy
import pytest
import numpy as np
from numpy.random import RandomState
from scipy.signal import find_peaks_cwt, find_peaks
from conftest import (
    assert_basis,
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_derivatives,
    assert_systems,
    water,
)
from ase import Atoms, geometry
from ase.build import molecule, bulk
from dscribe.descriptors import MBTR


# =============================================================================
# Utilities
default_k1 = {
    "geometry": {"function": "atomic_number"},
    "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 50},
    "weighting": {"function": "unity"}
}

default_k2 = {
    "geometry": {"function": "inverse_distance"},
    "grid": {"min": 0, "max": 1 / 0.7, "sigma": 0.1, "n": 50},
    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
}

default_k3 = {
    "geometry": {"function": "angle"},
    "grid": {"min": 0, "max": 180, "sigma": 2, "n": 50},
    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
}


def mbtr(**kwargs):
    """Returns a function that can be used to create a valid MBTR
    descriptor for a dataset.
    """
    def func(systems=None):
        species = set()
        for system in systems or []:
            species.update(system.get_atomic_numbers())
        final_kwargs = {
            "species": species,
            "geometry": default_k2["geometry"],
            "grid": default_k2["grid"],
            "weighting": default_k2["weighting"]
        }
        final_kwargs.update(kwargs)
        return MBTR(**final_kwargs)

    return func


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
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
    assert_parallellization(mbtr, n_jobs, flatten, sparse)


def test_no_system_modification():
    assert_no_system_modification(mbtr)


@pytest.mark.parametrize(
    "pbc, cell",
    [
        pytest.param([True, True, True], [5, 5, 5], id="Fully periodic system with cell"),
        pytest.param([False, False, False], None, id="Unperiodic system with no cell"),
        pytest.param([False, False, False], [0, 0, 0], id="Unperiodic system with no cell"),
        pytest.param([True, False, False], [5, 5, 5], id="Partially periodic system with cell"),
        pytest.param([True, True, True], [[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]], id="Fully periodic system with non-cubic cell"),
    ]
)
def test_systems(pbc, cell):
    assert_systems(mbtr(periodic=True), pbc, cell)


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
    assert_sparse(mbtr)


def test_symmetries():
    assert_symmetries(mbtr(), True, True, True)


# def test_derivatives():
#     assert_derivatives(mbtr(), 'numerical')


# =============================================================================
# Tests that are specific to this descriptor.
def test_exceptions():
    """Tests different invalid parameters that should raise an
    exception.
    """
    # Weighting needs to be provided for periodic system and terms k>1
    msg = "Periodic systems need to have a weighting function."
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            geometry=default_k2["geometry"],
            grid=default_k2["grid"],
            weighting={},
            periodic=True,
        )
    assert msg == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            geometry=default_k2["geometry"],
            grid=default_k2["grid"],
            weighting={"function": "unity"},
            periodic=True,
        )
    assert msg == str(excinfo.value)

    # Invalid weighting function
    msg = "Unknown weighting function."
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            geometry=default_k1["geometry"],
            grid=default_k1["grid"],
            weighting={"function": "exp", "threshold": 1, "scale": 1},
            periodic=True,
        )
    assert msg == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            geometry=default_k2["geometry"],
            grid=default_k2["grid"],
            weighting={"function": "none"},
            periodic=True,
        )
    assert msg == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            geometry=default_k3["geometry"],
            grid=default_k3["grid"],
            weighting={"function": "none"},
            periodic=True,
        )
    assert msg == str(excinfo.value)

    # Invalid geometry function
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            geometry={"function": "none"},
            grid={"min": 0, "max": 1, "n": 10, "sigma": 0.1},
            periodic=False,
        )
    msg = "Unknown geometry function."
    assert msg == str(excinfo.value)

    # Missing threshold
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        del setup["weighting"]["threshold"]
        MBTR(
            species=[1],
            geometry=setup["geometry"],
            grid=setup["grid"],
            weighting=setup["weighting"],
            periodic=True,
        )
    msg = "Missing value for 'threshold'."
    assert msg == str(excinfo.value)

    # Missing scale or r_cut
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        del setup["weighting"]["scale"]
        MBTR(
            species=[1],
            geometry=setup["geometry"],
            grid=setup["grid"],
            weighting=setup["weighting"],
            periodic=True,
        )
    msg = "Provide either 'scale' or 'r_cut'."
    assert msg == str(excinfo.value)

    # Both scale and r_cut provided
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        setup["weighting"]["scale"] = 1
        setup["weighting"]["r_cut"] = 1
        MBTR(
            species=[1],
            geometry=setup["geometry"],
            grid=setup["grid"],
            weighting=setup["weighting"],
            periodic=True,
        )
    msg = "Provide only 'scale' or 'r_cut', not both."
    assert msg == str(excinfo.value)

    # Term location not available
    desc = MBTR(
        species=[1],
        geometry=default_k2["geometry"],
        grid=default_k2["grid"],
        weighting=default_k2["weighting"],
        periodic=True,
    )
    with pytest.raises(ValueError) as excinfo:
        desc.get_location(("H"))
    msg = "Cannot retrieve the location for ('H'), as the used geometry function does not match the order k=1."
    assert msg == str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        desc.get_location(("H", "H", "H"))
    msg = "Cannot retrieve the location for ('H', 'H', 'H'), as the used geometry function does not match the order k=3."
    assert msg == str(excinfo.value)

    # Unknown normalization
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            geometry=default_k2["geometry"],
            grid=default_k2["grid"],
            weighting=default_k2["weighting"],
            normalization="l2_test",
            periodic=True,
        )
    msg = "Unknown normalization option."
    assert msg == str(excinfo.value)


@pytest.mark.parametrize(
    "setup, n_features",
    [
        pytest.param(default_k1, 2 * default_k1["grid"]["n"], id="K1"),
        pytest.param(default_k2, 2 * default_k2["grid"]["n"] * 1 / 2 * (2 + 1), id="K2"),
        pytest.param(default_k3, 2 * default_k3["grid"]["n"] * 1 / 2 * (2 + 1) * 2, id="K3"),
    ],
)
def test_number_of_features(setup, n_features):
    atomic_numbers = [1, 8]
    desc = MBTR(species=atomic_numbers, **setup)
    n_features = desc.get_number_of_features()
    assert n_features == n_features

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
        geometry={"function": "atomic_number"},
        grid={"min": start, "max": stop, "sigma": std, "n": n},
        normalize_gaussians = normalize_gaussians
    )
    system = water()
    y = desc.create(system)
    x = np.linspace(start, stop, n)

    # Find the location of the peaks
    h_loc = desc.get_location(["H"])
    peak1_x = np.searchsorted(x, 1)
    h_feat =  y[h_loc]
    peak1_y = h_feat[peak1_x]
    o_loc = desc.get_location(["O"])
    peak2_x = np.searchsorted(x, 8)
    o_feat =  y[o_loc]
    peak2_y = o_feat[peak2_x]

    # Check against the analytical value
    prefactor = 1 / np.sqrt(2 * np.pi) if normalize_gaussians else 1
    gaussian = (
        lambda x, mean: 1 / std * prefactor
        * np.exp(-((x - mean) ** 2) / (2 * std**2))
    )
    assert np.allclose(peak1_y, 2 * gaussian(1, 1), rtol=0, atol=0.001)
    assert np.allclose(peak2_y, gaussian(8, 8), rtol=0, atol=0.001)

    # Check the integral
    pdf = y[h_loc]
    dx = (stop - start) / (n - 1)
    sum_cum = np.sum(0.5 * dx * (pdf[:-1] + pdf[1:]))
    exp = 1 if normalize_gaussians else 1 / (1 / math.sqrt(2 * math.pi * std**2))
    assert np.allclose(sum_cum, 2*exp, rtol=0, atol=0.001)


@pytest.mark.parametrize(
    "system",
    [
        pytest.param(molecule("CO2"), id="CO2"),
        pytest.param(molecule("H2O"), id="H2O"),
    ],
)
def test_locations(system):
    """Tests that the function used to query combination locations in the output
    works.
    """
    species = ["H", "O", "C"]
    system_species = system.get_chemical_symbols()

    for k in range(1, 4):
        setup = globals()[f"default_k{k}"]
        desc = mbtr(**setup, periodic=False, species=species)([])
        feat = desc.create(system)
        combinations = itertools.combinations_with_replacement(species, k)
        for combination in combinations:
            loc = desc.get_location(combination)
            atom_combinations = itertools.permutations(system_species, k)
            exists = combination in atom_combinations
            if exists:
                assert feat[loc].sum() != 0
            else:
                assert feat[loc].sum() == 0

water_periodic = water()
water_periodic.set_pbc(True)
@pytest.mark.parametrize(
    "system,geometry,grid,weighting,periodic,peaks,prominence",
    [
        pytest.param(
            water(),
            {"function": "atomic_number"},
            {"min": 0, "max": 9, "sigma": 0.5, "n": 1000},
            None,
            False,
            [(("H"), [1], [2]), (("O"), [8], [1])],
            0.5,
            id="k1 finite"
        ),
        pytest.param(
            water_periodic,
            {"function": "atomic_number"},
            {"min": 0, "max": 9, "sigma": 0.5, "n": 1000},
            None,
            True,
            [(("H"), [1], [2]), (("O"), [8], [1])],
            0.5,
            id="k1 periodic"
        ),
        pytest.param(
            water(),
            {"function": "distance"},
            {"min": -1, "max": 3, "sigma": 0.5, "n": 1000},
            {"function": "unity"},
            False,
            [(("H", "H"), [1.4972204318527715], [1]), (("H", "O"), [0.95], [2])],
            0.5,
            id="k2 finite"
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
            {"function": "distance"},
            {"min": 0, "max": 10, "sigma": 0.5, "n": 1000},
            {"function": "exp", "scale": 0.8, "threshold": 1e-3},
            True,
            [(("H", "C"), [2, 8], np.exp(-0.8 * np.array([2, 8])))],
            0.001,
            id="k2 periodic"
        ),
        pytest.param(
            water(),
            {"function": "angle"},
            {"min": -10, "max": 180, "sigma": 5, "n": 2000},
            {"function": "unity"},
            False,
            [(("H", "H", "O"), [38], [2]), (("H", "O", "H"), [104], [1])],
            0.5,
            id="k3 finite"
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
            {"function": "angle"},
            {"min": 0, "max": 180, "sigma": 5, "n": 2000},
            {"function": "exp", "scale": 0.85, "threshold": 1e-3},
            True,
            [(("H", "H", "H"), [45, 90], [2 * np.exp(-0.85 * (2 + 2 * np.sqrt(2))), np.exp(-0.85 * (2 + 2 * np.sqrt(2)))])],
            0.01,
            id="k3 periodic"
        )
    ]
)
def test_peaks(system, geometry, grid, weighting, periodic, peaks, prominence):
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
    desc = MBTR(
        species=system.get_atomic_numbers(),
        geometry=geometry,
        grid=grid,
        weighting=weighting,
        normalize_gaussians=False,
        periodic=periodic,
        flatten=True,
        sparse=False,
    )
    features = desc.create(system)

    start = grid["min"]
    stop = grid["max"]
    n = grid["n"]
    x = np.linspace(start, stop, n)

    # Check that the correct peaks can be found
    for (location, peak_x, peak_y) in peaks:
        feat = features[desc.get_location(location)]

        # import matplotlib.pyplot as mpl 
        # mpl.plot(np.arange(len(feat)), feat)
        # mpl.show()

        peak_indices = find_peaks(feat, prominence=prominence)[0]
        assert len(peak_indices) > 0
        peak_locs = x[peak_indices]
        peak_ints = feat[peak_indices]
        assert np.allclose(peak_locs, peak_x, rtol=0, atol=5e-2)
        assert np.allclose(peak_ints, peak_y, rtol=0, atol=5e-2)

    # Check that everything else is zero
    for peak in peaks:
        features[desc.get_location(peak[0])] = 0
    assert features.sum() == 0


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(default_k2, id="K2"),
        pytest.param(default_k3, id="K3"),
    ]
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

    desc = MBTR(
        species=["H", "C"],
        **setup,
        periodic=True
    )

    # The resulting spectra should be identical
    spectra1 = desc.create(atoms)
    spectra2 = desc.create(atoms2)
    assert np.allclose(spectra1, spectra2, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "setup, normalization, norm",
    [
        pytest.param(default_k1, "l2", 1, id="K1"),
        pytest.param(default_k2, "l2", 1, id="K2"),
        pytest.param(default_k3, "l2", 1, id="K3"),
    ]
)
def test_normalization(setup, normalization, norm):
    """Tests that the normalization works correctly."""
    system = water()
    atomic_numbers = [1, 8]
    desc = MBTR(
        species=atomic_numbers,
        **setup,
        flatten=True,
        normalization=normalization
    )

    feat_normalized = desc.create(system)
    assert np.linalg.norm(feat_normalized) == pytest.approx(norm, abs=1e-8)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(
            {
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 100},
            },
            id="K1"
        ),
        pytest.param(
            {
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 200},
                "weighting": {
                    "function": "exp",
                    "scale": 1,
                    "threshold": 1e-3,
                },
            },
            id="K2"
        ),
        pytest.param(
            {
                "geometry": {"function": "cosine"},
                "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 200},
                "weighting": {
                    "function": "exp",
                    "scale": 1,
                    "threshold": 1e-3,
                }
            },
            id="K3"
        ),
    ]
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
        normalization="l2",
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
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 21},
            },
            id="K1"
        ),
        pytest.param(
            {
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 21},
                "weighting": {"function": "exp", "scale": 1, "threshold": 1e-4},
            },
            id="K2"
        ),
        pytest.param(
            {
                "geometry": {"function": "cosine"},
                "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 21},
                "weighting": {"function": "exp", "scale": 1, "threshold": 1e-4},
            },
            id="K3"
        ),
    ]
)
def test_periodic_images_1(setup):
    """Tests that periodic images are handled correctly."""
    decay = 1
    desc = MBTR(
        species=[1],
        periodic=True,
        **setup,
        normalization="l2",
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


def test_periodic_images_2():
    # Tests that the correct peak locations are present in a cubic periodic
    start = -1.1
    stop = 1.1
    n = 600
    desc = MBTR(
        species=["H"],
        periodic=True,
        geometry={"function": "cosine"},
        grid={"min": start, "max": stop, "sigma": 0.010, "n": n},
        weighting={"function": "exp", "scale": 1, "threshold": 1e-4},
        normalization="l2",
        flatten=True,
    )
    a = 2.2
    system = Atoms(
        cell=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        positions=[
            [0, 0, 0],
        ],
        symbols=["H"],
        pbc=True,
    )
    cubic_spectrum = desc.create(system)

    # from ase.visualize import view
    # view(system)
    # import matplotlib.pyplot as mpl 
    # mpl.plot(np.arange(len(cubic_spectrum)), cubic_spectrum)
    # mpl.show()

    x3 = np.linspace(start, stop, n)

    peak_ids = find_peaks_cwt(cubic_spectrum, [2])
    peak_locs = x3[peak_ids]

    assumed_peaks = np.cos(
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
    )
    assert np.allclose(peak_locs, assumed_peaks, rtol=0, atol=5 * np.pi / 180)

def test_periodic_images_3():
    # Tests that the correct peak locations are present in a system with a
    # non-cubic basis
    start = -1.0
    stop = 1.0
    n = 200
    desc = MBTR(
        species=["H"],
        periodic=True,
        geometry={"function": "cosine"},
        grid={"min": start, "max": stop, "sigma": 0.030, "n": n},
        weighting={"function": "exp", "scale": 1.5, "threshold": 1e-4},
        normalization="none",
        flatten=True,
        sparse=False,
    )
    a = 2.2
    angle = 30
    system = Atoms(
        cell=geometry.cellpar_to_cell([3 * a, a, a, angle, 90, 90]),
        positions=[
            [0, 0, 0],
        ],
        symbols=["H"],
        pbc=True,
    )
    tricl_spectrum = desc.create(system)
    x3 = np.linspace(start, stop, n)

    peak_ids = find_peaks_cwt(tricl_spectrum, [3])
    peak_locs = x3[peak_ids]

    angle = (6) / (np.sqrt(5) * np.sqrt(8))
    assumed_peaks = np.cos(np.array([180, 105, 75, 51.2, 30, 0]) * np.pi / 180)
    assert np.allclose(peak_locs, assumed_peaks, rtol=0, atol=5 * np.pi / 180)
