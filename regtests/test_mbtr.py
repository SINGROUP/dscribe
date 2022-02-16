import time
import pytest
import numpy as np
from numpy.random import RandomState
from scipy.signal import find_peaks_cwt, find_peaks
from conftest import (
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    water,
)
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
        for system in systems:
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


# =============================================================================
# Tests that are specific to this descriptor.
def test_exceptions():
    """Tests different invalid parameters that should raise an
    exception.
    """
    # Cannot create a sparse and non-flattened output.
    with pytest.raises(ValueError):
        MBTR(
            species=["H"],
            k1=default_k1,
            periodic=False,
            flatten=False,
            sparse=True,
        )


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

    import matplotlib.pyplot as mpl
    mpl.plot(np.arange(len(features)), features)
    mpl.show()

    # Check the H peaks
    h_feat = features[desc.get_location(("H"))]
    h_peak_indices = find_peaks(h_feat, prominence=1)[0]
    h_peak_locs = x[h_peak_indices]
    h_peak_ints = h_feat[h_peak_indices]
    assert np.allclose(h_peak_locs, [1], rtol=0, atol=1e-2)
    assert np.allclose(h_peak_ints, [2], rtol=0, atol=1e-2)

    # Check the O peaks
    # o_feat = features[desc.get_location(("O"))]
    # o_peak_indices = find_peaks(o_feat, prominence=1)[0]
    # o_peak_locs = x[o_peak_indices]
    # o_peak_ints = o_feat[o_peak_indices]
    # assert np.allclose(o_peak_locs, [8], rtol=0, atol=1e-2)
    # assert np.allclose(o_peak_ints, [1], rtol=0, atol=1e-2)

    # # Check that everything else is zero
    # features[desc.get_location(("H"))] = 0
    # features[desc.get_location(("O"))] = 0
    # assert features.sum() == 0
