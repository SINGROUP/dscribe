import time
import copy
import pytest
import numpy as np
from numpy.random import RandomState
from scipy.signal import find_peaks_cwt, find_peaks
from conftest import (
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_derivatives,
    water,
)
from ase import Atoms
from ase.build import molecule
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

    desc = MBTR(
        species=["H", "C"],
        k2={
            "geometry": {"function": "distance"},
            "grid": {"min": 0, "max": 10, "sigma": 0.5, "n": 1000},
            "weighting": {"function": "exp", "scale": 0.8, "threshold": 1e-3},
        },
        normalize_gaussians=False,
        periodic=True,
        flatten=True,
        sparse=False,
    )
    features = desc.create(atoms)
    x = desc.get_k2_axis()

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
