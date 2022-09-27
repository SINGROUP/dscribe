import time
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
            "k1": default_k1,
            "k2": default_k2,
            "k3": default_k3,
            "flatten": True,
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
        ([True, True, True], [5, 5, 5]),  # Fully periodic system with cell
        ([False, False, False], None),  # Unperiodic system with no cell
        ([False, False, False], [0, 0, 0]),  # Unperiodic system with no cell
        ([True, False, False], [5, 5, 5]),  # Partially periodic system with cell
        ([True, True, True], [[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]]),  # Fully periodic system with non-cubic cell
    ],
)
def test_systems(pbc, cell):
    assert_systems(mbtr(periodic=True), pbc, cell)


# def test_basis():
#     assert_basis(mbtr(periodic=True))


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
    # Cannot create a sparse and non-flattened output.
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            k1=default_k1,
            periodic=False,
            flatten=False,
            sparse=True,
        )
    msg = "Sparse, non-flattened output is currently not supported. If you want a non-flattened output, please specify sparse=False in the MBTR constructor."
    assert msg in str(excinfo.value)

    # Weighting needs to be provided for periodic system and terms k>1
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            k2={"geometry": default_k2["geometry"], "grid": default_k2["grid"]},
            periodic=True,
        )
    msg = "Periodic systems need to have a weighting function."
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            k2={
                "geometry": default_k2["geometry"],
                "grid": default_k2["grid"],
                "weighting": {"function": "unity"},
            },
            periodic=True,
        )
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            k3={"geometry": default_k3["geometry"], "grid": default_k3["grid"]},
            periodic=True,
        )
    assert msg in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=["H"],
            k3={
                "geometry": default_k3["geometry"],
                "grid": default_k3["grid"],
                "weighting": {"function": "unity"},
            },
            periodic=True,
        )
    assert msg in str(excinfo.value)

    # Invalid weighting function
    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k1={
                "geometry": default_k1["geometry"],
                "grid": default_k1["grid"],
                "weighting": {"function": "none"},
            },
            periodic=True,
        )
    msg = "Unknown weighting function specified for k1."
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k2={
                "geometry": default_k2["geometry"],
                "grid": default_k2["grid"],
                "weighting": {"function": "none"},
            },
            periodic=True,
        )
    msg = "Unknown weighting function specified for k2."
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k3={
                "geometry": default_k3["geometry"],
                "grid": default_k3["grid"],
                "weighting": {"function": "none"},
            },
            periodic=True,
        )
    msg = "Unknown weighting function specified for k3."
    assert msg in str(excinfo.value)

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
    msg = "Unknown geometry function specified for k1."
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k2={
                "geometry": {"function": "none"},
                "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1},
            },
            periodic=False,
        )
    msg = "Unknown geometry function specified for k2."
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MBTR(
            species=[1],
            k3={
                "geometry": {"function": "none"},
                "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1},
            },
            periodic=False,
        )
    msg = "Unknown geometry function specified for k3."
    assert msg in str(excinfo.value)

    # Missing threshold
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        del setup["weighting"]["threshold"]
        MBTR(
            species=[1],
            k2=setup,
            periodic=True,
        )
    msg = "Missing value for 'threshold'"
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k3)
        del setup["weighting"]["threshold"]
        MBTR(
            species=[1],
            k3=setup,
            periodic=True,
        )
    assert msg in str(excinfo.value)

    # Missing scale or r_cut
    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k2)
        del setup["weighting"]["scale"]
        MBTR(
            species=[1],
            k2=setup,
            periodic=True,
        )
    msg = "Provide either 'scale' or 'r_cut'."
    assert msg in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        setup = copy.deepcopy(default_k3)
        del setup["weighting"]["scale"]
        MBTR(
            species=[1],
            k3=setup,
            periodic=True,
        )
    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "k1, k2, k3",
    [
        (True, False, False),  # K1
        (False, True, False),  # K2
        (False, False, True),  # K3
        (True, True, False),  # K1 + K2
        (True, False, True),  # K1 + K3
        (False, True, True),  # K2 + K3
        (True, True, True),  # K1 + K2 + K3
    ],
)
def test_number_of_features(k1, k2, k3):
    atomic_numbers = [1, 8]
    n_elem = len(atomic_numbers)

    desc = MBTR(
        species=atomic_numbers,
        k1=default_k1 if k1 else None,
        k2=default_k2 if k2 else None,
        k3=default_k3 if k3 else None,
        flatten=True,
    )
    n_features = desc.get_number_of_features()

    expected_k1 = n_elem * default_k1["grid"]["n"]
    expected_k2 = n_elem * default_k2["grid"]["n"] * 1 / 2 * (n_elem + 1)
    expected_k3 = n_elem * default_k3["grid"]["n"] * 1 / 2 * (n_elem + 1) * n_elem

    assert n_features == (expected_k1 if k1 else 0) + (expected_k2 if k2 else 0) + (expected_k3 if k3 else 0)


@pytest.mark.parametrize(
    "normalize_gaussians",
    [
        (True),
        (False),
    ],
)
def test_gaussian_distribution(normalize_gaussians, H2O):
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
            "grid": {"min": start, "max": stop, "sigma": std, "n": n} 
        },
        normalize_gaussians = normalize_gaussians
    )
    y = desc.create(H2O)
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


def test_locations_k1():
    """Tests that the function used to query combination locations for k=1
    in the output works.
    """
    CO2 = molecule("CO2")
    H2O = molecule("H2O")

    desc = mbtr(periodic = False, species = ["H", "O", "C"])([])

    co2_out = desc.create(CO2)[:]
    h2o_out = desc.create(H2O)[:]

    loc_h = desc.get_location(("H"))
    loc_c = desc.get_location(("C"))
    loc_o = desc.get_location(("O"))

    # H
    assert co2_out[loc_h].sum() == 0
    assert h2o_out[loc_h].sum() != 0

    # C
    assert co2_out[loc_c].sum() != 0
    assert h2o_out[loc_c].sum() == 0

    # O
    assert co2_out[loc_o].sum() != 0
    assert h2o_out[loc_o].sum() != 0

def test_locations_k2():
    """Tests that the function used to query combination locations for k=2
    in the output works.
    """
    CO2 = molecule("CO2")
    H2O = molecule("H2O")

    desc = mbtr(periodic = False, species = ["H", "O", "C"])([])

    co2_out = desc.create(CO2)[:]
    h2o_out = desc.create(H2O)[:]

    loc_hh = desc.get_location(("H", "H"))
    loc_hc = desc.get_location(("H", "C"))
    loc_ho = desc.get_location(("H", "O"))
    loc_co = desc.get_location(("C", "O"))
    loc_cc = desc.get_location(("C", "C"))
    loc_oo = desc.get_location(("O", "O"))

    # H-H
    assert co2_out[loc_hh].sum() == 0
    assert h2o_out[loc_hh].sum() != 0

    # H-C
    assert co2_out[loc_hc].sum() == 0
    assert h2o_out[loc_hc].sum() == 0

    # H-O
    assert co2_out[loc_ho].sum() == 0
    assert h2o_out[loc_ho].sum() != 0

    # C-O
    assert co2_out[loc_co].sum() != 0
    assert h2o_out[loc_co].sum() == 0

    # C-C
    assert co2_out[loc_cc].sum() == 0
    assert h2o_out[loc_cc].sum() == 0

    # O-O
    assert co2_out[loc_oo].sum() != 0
    assert h2o_out[loc_oo].sum() == 0


def test_locations_k3():
    """Tests that the function used to query combination locations for k=2
    in the output works.
    """
    CO2 = molecule("CO2")
    H2O = molecule("H2O")

    desc = mbtr(periodic = False, species = ["H", "O", "C"])([])

    co2_out = desc.create(CO2)[:]
    h2o_out = desc.create(H2O)[:]

    loc_hhh = desc.get_location(("H", "H", "H"))
    loc_hho = desc.get_location(("H", "H", "O"))
    loc_hoo = desc.get_location(("H", "O", "O"))
    loc_hoh = desc.get_location(("H", "O", "H"))
    loc_ooo = desc.get_location(("O", "O", "O"))
    loc_ooh = desc.get_location(("O", "O", "H"))
    loc_oho = desc.get_location(("O", "H", "O"))
    loc_ohh = desc.get_location(("O", "H", "H"))

    # H-H-H
    assert co2_out[loc_hhh].sum() == 0
    assert h2o_out[loc_hhh].sum() == 0

    # H-H-O
    assert co2_out[loc_hho].sum() == 0
    assert h2o_out[loc_hho].sum() != 0

    # H-O-O
    assert co2_out[loc_hoo].sum() == 0
    assert h2o_out[loc_hoo].sum() == 0

    # H-O-H
    assert co2_out[loc_hoh].sum() == 0
    assert h2o_out[loc_hoh].sum() != 0

    # O-O-O
    assert co2_out[loc_ooo].sum() == 0
    assert h2o_out[loc_ooo].sum() == 0

    # O-O-H
    assert co2_out[loc_ooh].sum() == 0
    assert h2o_out[loc_ooh].sum() == 0

    # O-H-O
    assert co2_out[loc_oho].sum() == 0
    assert h2o_out[loc_oho].sum() == 0

    # O-H-H
    assert co2_out[loc_ohh].sum() == 0
    assert h2o_out[loc_ohh].sum() != 0


def test_k1_peaks_finite():
    """Tests the correct peak locations and intensities are found for the
    k=1 term.
    """
    system = water()
    k1 = {
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 0, "max": 9, "sigma": 0.5, "n": 1000},
    }
    desc = MBTR(
        species=[1, 8],
        k1=k1,
        normalize_gaussians=False,
        periodic=False,
        flatten=True,
        sparse=False,
    )
    features = desc.create(system)

    start = k1["grid"]["min"]
    stop = k1["grid"]["max"]
    n = k1["grid"]["n"]
    x = np.linspace(start, stop, n)

    # Check the H peaks
    h_feat = features[desc.get_location(("H"))]
    h_peak_indices = find_peaks(h_feat, prominence=1)[0]
    h_peak_locs = x[h_peak_indices]
    h_peak_ints = h_feat[h_peak_indices]
    assert np.allclose(h_peak_locs, [1], rtol=0, atol=1e-2)
    assert np.allclose(h_peak_ints, [2], rtol=0, atol=1e-2)

    # Check the O peaks
    o_feat = features[desc.get_location(("O"))]
    o_peak_indices = find_peaks(o_feat, prominence=1)[0]
    o_peak_locs = x[o_peak_indices]
    o_peak_ints = o_feat[o_peak_indices]
    assert np.allclose(o_peak_locs, [8], rtol=0, atol=1e-2)
    assert np.allclose(o_peak_ints, [1], rtol=0, atol=1e-2)

    # Check that everything else is zero
    features[desc.get_location(("H"))] = 0
    features[desc.get_location(("O"))] = 0
    assert features.sum() == 0

def test_k2_peaks_finite():
    """Tests the correct peak locations and intensities are found for the k=2
    term in finite systems.
    """
    system = water()
    k2 = {
        "geometry": {"function": "distance"},
        "grid": {"min": -1, "max": 3, "sigma": 0.5, "n": 1000},
        "weighting": {"function": "unity"},
    }
    desc = MBTR(
        species=[1, 8],
        k2=k2,
        normalize_gaussians=False,
        periodic=False,
        flatten=True,
        sparse=False,
    )
    features = desc.create(system)

    pos = system.get_positions()
    start = k2["grid"]["min"]
    stop = k2["grid"]["max"]
    n = k2["grid"]["n"]
    x = np.linspace(start, stop, n)

    # Check the H-H peaks
    hh_feat = features[desc.get_location(("H", "H"))]
    hh_peak_indices = find_peaks(hh_feat, prominence=0.5)[0]
    hh_peak_locs = x[hh_peak_indices]
    hh_peak_ints = hh_feat[hh_peak_indices]
    assert len(hh_peak_locs) > 0
    assert np.allclose(hh_peak_locs, [np.linalg.norm(pos[0] - pos[2])], rtol=0, atol=1e-2)
    assert np.allclose(hh_peak_ints, [1], rtol=0, atol=1e-2)

    # Check the O-H peaks
    ho_feat = features[desc.get_location(("H", "O"))]
    ho_peak_indices = find_peaks(ho_feat, prominence=0.5)[0]
    ho_peak_locs = x[ho_peak_indices]
    ho_peak_ints = ho_feat[ho_peak_indices]
    assert len(ho_peak_locs) > 0
    assert np.allclose(ho_peak_locs, np.linalg.norm(pos[0] - pos[1]), rtol=0, atol=1e-2)
    assert np.allclose(ho_peak_ints, [2], rtol=0, atol=1e-2)

    # Check that everything else is zero
    features[desc.get_location(("H", "H"))] = 0
    features[desc.get_location(("H", "O"))] = 0
    assert features.sum() == 0

def test_k3_peaks_finite():
    """Tests that all the correct angles are present in finite systems.
    There should be n*(n-1)*(n-2)/2 unique angles where the division by two
    gets rid of duplicate angles.
    """
    system = water()
    k3 = {
        "geometry": {"function": "angle"},
        "grid": {"min": -10, "max": 180, "sigma": 5, "n": 2000},
        "weighting": {"function": "unity"},
    }
    desc = MBTR(
        species=["H", "O"],
        k3=k3,
        normalize_gaussians=False,
        periodic=False,
        flatten=True,
        sparse=False,
    )
    features = desc.create(system)

    start = k3["grid"]["min"]
    stop = k3["grid"]["max"]
    n = k3["grid"]["n"]
    x = np.linspace(start, stop, n)

    # Check the H-H-O peaks
    hho_assumed_locs = np.array([38])
    hho_assumed_ints = np.array([2])
    hho_feat = features[desc.get_location(("H", "H", "O"))]
    hho_peak_indices = find_peaks(hho_feat, prominence=0.5)[0]
    hho_peak_locs = x[hho_peak_indices]
    hho_peak_ints = hho_feat[hho_peak_indices]

    assert np.allclose(hho_peak_locs, hho_assumed_locs, rtol=0, atol=5e-2)
    assert np.allclose(hho_peak_ints, hho_assumed_ints, rtol=0, atol=5e-2)

    # Check the H-O-H peaks
    hoh_assumed_locs = np.array([104])
    hoh_assumed_ints = np.array([1])
    hoh_feat = features[desc.get_location(("H", "O", "H"))]
    hoh_peak_indices = find_peaks(hoh_feat, prominence=0.5)[0]
    hoh_peak_locs = x[hoh_peak_indices]
    hoh_peak_ints = hoh_feat[hoh_peak_indices]
    assert np.allclose(hoh_peak_locs, hoh_assumed_locs, rtol=0, atol=5e-2)
    assert np.allclose(hoh_peak_ints, hoh_assumed_ints, rtol=0, atol=5e-2)

    # Check that everything else is zero
    features[desc.get_location(("H", "H", "O"))] = 0
    features[desc.get_location(("H", "O", "H"))] = 0
    assert features.sum() == 0


def test_k2_peaks_periodic():
    """Tests the correct peak locations and intensities are found for the
    k=2 term in periodic systems.
    """
    atoms = Atoms(
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
    )

    k2 = {
        "geometry": {"function": "distance"},
        "grid": {"min": 0, "max": 10, "sigma": 0.5, "n": 1000},
        "weighting": {"function": "exp", "scale": 0.8, "threshold": 1e-3},
    }
    desc = MBTR(
        species=["H", "C"],
        k2=k2,
        normalize_gaussians=False,
        periodic=True,
        flatten=True,
        sparse=False,
    )
    features = desc.create(atoms)

    start = k2["grid"]["min"]
    stop = k2["grid"]["max"]
    n = k2["grid"]["n"]
    x = np.linspace(start, stop, n)

    # Calculate assumed locations and intensities.
    assumed_locs = np.array([2, 8])
    assumed_ints = np.exp(-0.8 * np.array([2, 8]))
    assumed_ints[0] *= 2  # There are two periodic distances at 2Å
    assumed_ints[
        0
    ] /= (
        2  # The periodic distances ar halved because they belong to different cells
    )

    # Check the H-C peaks
    hc_feat = features[desc.get_location(("H", "C"))]
    hc_peak_indices = find_peaks(hc_feat, prominence=0.001)[0]
    hc_peak_locs = x[hc_peak_indices]
    hc_peak_ints = hc_feat[hc_peak_indices]
    assert np.allclose(hc_peak_locs, assumed_locs, rtol=0, atol=1e-2)
    assert np.allclose(hc_peak_ints, assumed_ints, rtol=0, atol=1e-2)

    # Check that everything else is zero
    features[desc.get_location(("H", "C"))] = 0
    assert features.sum() == 0


@pytest.mark.parametrize(
    "k2, k3",
    [
        (True, False),  # K2
        (False, True),  # K3
    ]
)
def test_periodic_translation(k2, k3):
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
        k2=default_k2 if k2 else None,
        k3=default_k3 if k3 else None,
        periodic=True
    )

    # The resulting spectra should be identical
    spectra1 = desc.create(atoms)
    spectra2 = desc.create(atoms2)
    assert np.allclose(spectra1, spectra2, rtol=0, atol=1e-10)


def test_k3_peaks_periodic():
    scale = 0.85
    k3 = {
        "geometry": {"function": "angle"},
        "grid": {"min": 0, "max": 180, "sigma": 5, "n": 2000},
        "weighting": {"function": "exp", "scale": scale, "threshold": 1e-3},
    }
    desc = MBTR(
        species=["H"],
        k3=k3,
        normalize_gaussians=False,
        periodic=True,
        flatten=True,
        sparse=False,
    )

    atoms = Atoms(
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
    )
    features = desc.create(atoms)
    start = k3["grid"]["min"]
    stop = k3["grid"]["max"]
    n = k3["grid"]["n"]
    x = np.linspace(start, stop, n)

    # Calculate assumed locations and intensities.
    assumed_locs = np.array([45, 90])
    dist = 2 + 2 * np.sqrt(2)  # The total distance around the three atoms
    weight = np.exp(-scale * dist)
    assumed_ints = np.array([4 * weight, 2 * weight])
    assumed_ints /= (
        2  # The periodic distances ar halved because they belong to different cells
    )

    # Check the H-H-H peaks
    hhh_feat = features[desc.get_location(("H", "H", "H"))]
    hhh_peak_indices = find_peaks(hhh_feat, prominence=0.01)[0]
    hhh_peak_locs = x[hhh_peak_indices]
    hhh_peak_ints = hhh_feat[hhh_peak_indices]
    assert np.allclose(hhh_peak_locs, assumed_locs, rtol=0, atol=1e-1)
    assert np.allclose(hhh_peak_ints, assumed_ints, rtol=0, atol=1e-1)

    # Check that everything else is zero
    features[desc.get_location(("H", "H", "H"))] = 0
    assert features.sum() == 0


@pytest.mark.parametrize(
    "k1, k2, k3, normalization, norm",
    [
        (True, False, False, "l2", 1),       # K1
        (True, False, False, "l2_each", 1),  # K1
        (False, True, False, "l2", 1),       # K2
        (False, True, False, "l2_each", 1),  # K2
        (False, False, True, "l2", 1),       # K3
        (False, False, True, "l2_each", 1),  # K3
        (True, True, True, "l2", 1),        # K1 + K2 + K3
        (True, True, True, "l2_each", np.sqrt(3)),   # K1 + K2 + K3
    ]
)
def test_normalization(k1, k2, k3, normalization, norm):
    """Tests that the normalization works correctly."""
    system = water()
    atomic_numbers = [1, 8]
    desc = MBTR(
        species=atomic_numbers,
        k1=default_k1 if k1 else None,
        k2=default_k2 if k2 else None,
        k3=default_k3 if k3 else None,
        flatten=True,
        normalization=normalization
    )

    feat_normalized = desc.create(system)
    assert np.linalg.norm(feat_normalized) == pytest.approx(norm, abs=1e-8)


# def test_periodic_supercell_similarity():
#     """Tests that the output spectrum of various supercells of the same
#     crystal is identical after it is normalized.
#     """
#     decay = 1
#     desc = MBTR(
#         species=["H"],
#         periodic=True,
#         k1={
#             "geometry": {"function": "atomic_number"},
#             "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 100},
#         },
#         k2={
#             "geometry": {"function": "inverse_distance"},
#             "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 200},
#             "weighting": {
#                 "function": "exp",
#                 "scale": decay,
#                 "threshold": 1e-3,
#             },
#         },
#         k3={
#             "geometry": {"function": "cosine"},
#             "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 200},
#             "weighting": {
#                 "function": "exp",
#                 "scale": decay,
#                 "threshold": 1e-3,
#             },
#         },
#         flatten=True,
#         sparse=False,
#         normalization="l2_each",
#     )

#     # Create various supercells for the FCC structure
#     a1 = bulk("H", "fcc", a=2.0)  # Primitive
#     a2 = a1 * [2, 2, 2]  # Supercell
#     a3 = bulk("H", "fcc", a=2.0, orthorhombic=True)  # Orthorhombic
#     a4 = bulk("H", "fcc", a=2.0, cubic=True)  # Conventional cubic

#     output = desc.create([a1, a2, a3, a4])

#     # Test for equality
#     assert np.allclose(output[0, :], output[0, :], atol=1e-5, rtol=0)
#     assert np.allclose(output[0, :], output[1, :], atol=1e-5, rtol=0)
#     assert np.allclose(output[0, :], output[2, :], atol=1e-5, rtol=0)
#     assert np.allclose(output[0, :], output[3, :], atol=1e-5, rtol=0)


# def test_periodic_images():
#     """Tests that periodic images are handled correctly."""
#     decay = 1
#     desc = MBTR(
#         species=[1],
#         periodic=True,
#         k1={
#             "geometry": {"function": "atomic_number"},
#             "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 21},
#         },
#         k2={
#             "geometry": {"function": "inverse_distance"},
#             "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 21},
#             "weighting": {"function": "exp", "scale": decay, "threshold": 1e-4},
#         },
#         k3={
#             "geometry": {"function": "cosine"},
#             "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 21},
#             "weighting": {"function": "exp", "scale": decay, "threshold": 1e-4},
#         },
#         normalization="l2_each",  # This normalizes the spectrum
#         flatten=True,
#     )

#     # Tests that a system has the same spectrum as the supercell of the same
#     # system.
#     molecule = Atoms(
#         cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
#         positions=[
#             [0, 0, 0],
#         ],
#         symbols=["H"],
#     )

#     a = 1.5
#     molecule.set_cell([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
#     molecule.set_pbc(True)
#     cubic_cell = desc.create(molecule)
#     suce = molecule * (2, 1, 1)
#     cubic_suce = desc.create(suce)

#     diff = abs(np.sum(cubic_cell - cubic_suce))
#     cubic_sum = abs(np.sum(cubic_cell))
#     assert diff / cubic_sum < 0.05  # A 5% error is tolerated

#     # Same test but for triclinic cell
#     molecule.set_cell([[0.0, 2.0, 1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 0.0]])

#     triclinic_cell = desc.create(molecule)
#     suce = molecule * (2, 1, 1)
#     triclinic_suce = desc.create(suce)

#     diff = abs(np.sum(triclinic_cell - triclinic_suce))
#     tricl_sum = abs(np.sum(triclinic_cell))
#     assert diff / tricl_sum < 0.05

#     # Testing that the same crystal, but different unit cells will have a
#     # similar spectrum when they are normalized. There will be small differences
#     # in the shape (due to not double counting distances)
#     a1 = bulk("H", "fcc", a=2.0)
#     a2 = bulk("H", "fcc", a=2.0, orthorhombic=True)
#     a3 = bulk("H", "fcc", a=2.0, cubic=True)

#     triclinic_cell = desc.create(a1)
#     orthorhombic_cell = desc.create(a2)
#     cubic_cell = desc.create(a3)

#     diff1 = abs(np.sum(triclinic_cell - orthorhombic_cell))
#     diff2 = abs(np.sum(triclinic_cell - cubic_cell))
#     tricl_sum = abs(np.sum(triclinic_cell))
#     assert diff1 / tricl_sum < 0.05
#     assert diff2 / tricl_sum < 0.05

#     # Tests that the correct peak locations are present in a cubic periodic
#     desc = MBTR(
#         species=["H"],
#         periodic=True,
#         k3={
#             "geometry": {"function": "cosine"},
#             "grid": {"min": -1.1, "max": 1.1, "sigma": 0.010, "n": 600},
#             "weighting": {"function": "exp", "scale": decay, "threshold": 1e-4},
#         },
#         normalization="l2_each",  # This normalizes the spectrum
#         flatten=True,
#     )
#     a = 2.2
#     system = Atoms(
#         cell=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
#         positions=[
#             [0, 0, 0],
#         ],
#         symbols=["H"],
#         pbc=True,
#     )
#     cubic_spectrum = desc.create(system)
#     x3 = desc.get_k3_axis()

#     peak_ids = find_peaks_cwt(cubic_spectrum, [2])
#     peak_locs = x3[peak_ids]

#     assumed_peaks = np.cos(
#         np.array(
#             [
#                 180,
#                 90,
#                 np.arctan(np.sqrt(2)) * 180 / np.pi,
#                 45,
#                 np.arctan(np.sqrt(2) / 2) * 180 / np.pi,
#                 0,
#             ]
#         )
#         * np.pi
#         / 180
#     )
#     assert np.allclose(peak_locs, assumed_peaks, rtol=0, atol=5 * np.pi / 180)

#     # Tests that the correct peak locations are present in a system with a
#     # non-cubic basis
#     desc = MBTR(
#         species=["H"],
#         periodic=True,
#         k3={
#             "geometry": {"function": "cosine"},
#             "grid": {"min": -1.0, "max": 1.0, "sigma": 0.030, "n": 200},
#             "weighting": {"function": "exp", "scale": 1.5, "threshold": 1e-4},
#         },
#         normalization="l2_each",  # This normalizes the spectrum
#         flatten=True,
#         sparse=False,
#     )
#     a = 2.2
#     angle = 30
#     system = Atoms(
#         cell=geometry.cellpar_to_cell([3 * a, a, a, angle, 90, 90]),
#         positions=[
#             [0, 0, 0],
#         ],
#         symbols=["H"],
#         pbc=True,
#     )
#     tricl_spectrum = desc.create(system)
#     x3 = desc.get_k3_axis()

#     peak_ids = find_peaks_cwt(tricl_spectrum, [3])
#     peak_locs = x3[peak_ids]

#     angle = (6) / (np.sqrt(5) * np.sqrt(8))
#     assumed_peaks = np.cos(np.array([180, 105, 75, 51.2, 30, 0]) * np.pi / 180)
#     assert np.allclose(peak_locs, assumed_peaks, rtol=0, atol=5 * np.pi / 180)
