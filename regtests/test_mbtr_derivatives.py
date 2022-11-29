import pytest
import numpy as np
from conftest import (
    water,
    assert_derivatives,
)
from dscribe.descriptors import MBTR


def k2_dict(geometry_function, weighting_function):
    d = {}
    if geometry_function == "inverse_distance":
        d["geometry"] = {"function": "inverse_distance"}
        d["grid"] = {"min": 0, "max": 1.0, "sigma": 0.02, "n": 100}
    else:
        d["geometry"] = {"function": "distance"}
        d["grid"] = {"min": 0, "max": 10.0, "sigma": 0.2, "n": 100}

    if weighting_function == "exp":
        d["weighting"] = {"function": "exp", "r_cut": 10.0, "threshold": 1e-3}

    return d


def k3_dict(weighting_function):
    d = {}
    d["geometry"] = {"function": "cosine"}
    d["grid"] = {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 100}

    if weighting_function == "exp":
        d["weighting"] = {"function": "exp", "scale": 1.0, "threshold": 1e-3}

    return d


def mbtr(**kwargs):
    """Returns a function that can be used to create a valid MBTR
    descriptor for a dataset.
    """

    def func(systems=None):
        species = list(set().union(*[s.get_atomic_numbers() for s in systems]))

        final_kwargs = {
            "species": species,
            "normalization": "none",
            "periodic": True,
            "sparse": False,
            "flatten": True,
            "k1": {
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 0, "max": 9, "sigma": 0.1, "n": 100},
            },
            "k2": k2_dict("inverse_distance", "exp"),
            "k3": k3_dict("exp"),
        }
        final_kwargs.update(kwargs)

        return MBTR(**final_kwargs)

    return func


def assert_derivatives_analytical(descriptor_func):
    """Test analytical values against a numerical python implementation."""
    system = water()
    system.pbc = [True, True, True]
    h = 0.0001
    n_atoms = len(system)
    n_comp = 3
    descriptor = descriptor_func([system])

    n_features = descriptor.get_number_of_features()
    derivatives_python = np.zeros((n_atoms, n_comp, n_features))
    d0 = descriptor.create(system)
    coeffs = [-1.0 / 2.0, 1.0 / 2.0]
    deltas = [-1.0, 1.0]
    for i_atom in range(len(system)):
        for i_comp in range(3):
            for i_stencil in range(2):
                system_disturbed = system.copy()
                i_pos = system_disturbed.get_positions()
                i_pos[i_atom, i_comp] += h * deltas[i_stencil]
                system_disturbed.set_positions(i_pos)
                d1 = descriptor.create(system_disturbed)
                derivatives_python[i_atom, i_comp, :] += coeffs[i_stencil] * d1 / h

    # Calculate analytical derivatives
    derivatives_analytical, d_cpp = descriptor.derivatives(system, method="analytical")

    # Compare descriptor values
    assert np.allclose(d0, d_cpp, atol=1e-6)

    # Compare derivative values. The check is not very strict because there
    # is no guarantee that the numerical derivatives are accurate
    max_error = np.max(np.abs(derivatives_python - derivatives_analytical))
    max_value = np.max(np.abs(derivatives_python))
    assert max_error / max_value < 1e-2


@pytest.mark.parametrize(
    "normalization, periodic, k2, k3, method",
    [
        (
            "none",
            True,
            k2_dict("inverse_distance", "exp"),
            k3_dict("exp"),
            "analytical",
        ),
        (
            "none",
            False,
            k2_dict("inverse_distance", "none"),
            k3_dict("none"),
            "analytical",
        ),
        (
            "none",
            True,
            k2_dict("distance", "exp"),
            k3_dict("exp"),
            "analytical",
        ),
        (
            "n_atoms",
            True,
            k2_dict("inverse_distance", "exp"),
            k3_dict("exp"),
            "analytical",
        ),
    ],
)
def test_derivatives(normalization, periodic, k2, k3, method):
    mbtr_func = mbtr(normalization=normalization, periodic=periodic, k2=k2, k3=k3)
    assert_derivatives(mbtr_func, method)
    if method == "analytical":
        assert_derivatives_analytical(mbtr_func)
