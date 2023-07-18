import pytest
from pathlib import Path
import itertools
import numpy as np
from ase import Atoms
from conftest import (
    assert_n_features,
    assert_dtype,
    assert_cell,
    assert_no_system_modification,
    assert_sparse,
    assert_basis,
    assert_parallellization,
    assert_symmetries,
    assert_centers,
    assert_periodic,
    assert_derivatives,
    assert_derivatives_exclude,
    assert_derivatives_include,
    get_simple_finite,
)
from dscribe.descriptors import SOAP


# =============================================================================
# Utilities
default_r_cut = 3
default_n_max = 3
default_l_max = 6
folder = Path(__file__).parent


def soap(**kwargs):
    """Returns a function that can be used to create a valid ACSF
    descriptor for a dataset.
    """

    def func(systems=None):
        species = set()
        for system in systems:
            species.update(system.get_atomic_numbers())
        final_kwargs = {
            "species": species,
            "r_cut": default_r_cut,
            "n_max": default_n_max,
            "l_max": default_l_max,
        }
        final_kwargs.update(kwargs)
        return SOAP(**final_kwargs)

    return func


def get_soap_default_setup():
    """Returns an atomic system and SOAP parameters which are ideal for most
    tests.
    """
    # Calculate the numerical power spectrum
    system = Atoms(
        positions=[
            [0.0, 0.0, 0.0],
            [-0.3, 0.5, 0.4],
        ],
        symbols=["H", "C"],
    )

    centers = [
        [0, 0, 0],
        [1 / 3, 1 / 3, 1 / 3],
        [2 / 3, 2 / 3, 2 / 3],
    ]

    soap_arguments = {
        "n_max": 2,
        "l_max": 2,
        "r_cut": 2.0,
        "sigma": 0.55,
        "species": ["H", "C"],
        "crossover": True,
    }
    return [system, centers, soap_arguments]


def get_soap_gto_l_max_setup():
    """Returns an atomic system and SOAP parameters which are ideal for quickly
    testing the correctness of SOAP values with large l_max and the GTO basis.
    Minimizing the computation is important because the calculating the
    numerical benchmark values is very expensive.
    """
    # We need a lot of atoms in order to make the values at large l-values to
    # be above numerical precision and in order to have good distribution for
    # all l-components.
    x, y, z = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))
    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(float)
    rng = np.random.RandomState(42)
    positions += rng.random((27, 3)) - 0.5

    # One atom should be exactly at origin because it is an exceptional
    # location that needs to be tested.
    positions[13, :] = [0, 0, 0]

    system = Atoms(
        positions=positions,
        symbols=len(positions) * ["H"],
    )

    centers = [[0, 0, 0]]

    # Making sigma small enough ensures that the smaller l-components are not
    # screened by big fluffy gaussians.
    soap_arguments = {
        "n_max": 1,
        "l_max": 20,
        "r_cut": 2.0,
        "sigma": 0.1,
        "species": ["H"],
        "crossover": False,
    }
    return (system, centers, soap_arguments)


def get_soap_polynomial_l_max_setup():
    """Returns an atomic system and SOAP parameters which are ideal for quickly
    testing the correctness of SOAP values with large l_max and the polynomial
    basis.  Minimizing the computation is important because the calculating the
    numerical benchmark values is very expensive.
    """
    # We need a lot of atoms in order to make the values at large l-values to
    # be above numerical precision and in order to have good distribution for
    # all l-components.
    x, y, z = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))
    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(float)
    rng = np.random.RandomState(42)
    positions += rng.random((27, 3)) - 0.5

    # One atom should be exactly at origin because it is an exceptional
    # location that needs to be tested.
    positions[13, :] = [0, 0, 0]

    system = Atoms(
        positions=positions,
        symbols=len(positions) * ["H"],
    )

    centers = [[0, 0, 0]]

    # Making sigma small enough ensures that the smaller l-components are not
    # screened by big fluffy gaussians.
    soap_arguments = {
        "n_max": 1,
        "l_max": 9,
        "r_cut": 2.0,
        "sigma": 0.1,
        "species": ["H"],
        "crossover": False,
    }
    return (system, centers, soap_arguments)


def get_power_spectrum(coeffs, crossover=True, average="off"):
    """Given the expansion coefficients, returns the power spectrum."""
    numerical_power_spectrum = []
    shape = coeffs.shape
    n_centers = 1 if average != "off" else shape[0]
    n_species = shape[1]
    n_max = shape[2]
    l_max = shape[3] - 1
    for i in range(n_centers):
        i_spectrum = []
        for zi in range(n_species):
            for zj in range(zi, n_species if crossover else zi + 1):
                if zi == zj:
                    for l in range(l_max + 1):
                        for ni in range(n_max):
                            for nj in range(ni, n_max):
                                if average == "inner":
                                    value = np.dot(
                                        coeffs[:, zi, ni, l, :].mean(axis=0),
                                        coeffs[:, zj, nj, l, :].mean(axis=0),
                                    )
                                else:
                                    value = np.dot(
                                        coeffs[i, zi, ni, l, :],
                                        coeffs[i, zj, nj, l, :],
                                    )
                                prefactor = np.pi * np.sqrt(8 / (2 * l + 1))
                                value *= prefactor
                                i_spectrum.append(value)
                else:
                    for l in range(l_max + 1):
                        for ni in range(n_max):
                            for nj in range(n_max):
                                if average == "inner":
                                    value = np.dot(
                                        coeffs[:, zi, ni, l, :].mean(axis=0),
                                        coeffs[:, zj, nj, l, :].mean(axis=0),
                                    )
                                else:
                                    value = np.dot(
                                        coeffs[i, zi, ni, l, :],
                                        coeffs[i, zj, nj, l, :],
                                    )
                                prefactor = np.pi * np.sqrt(8 / (2 * l + 1))
                                value *= prefactor
                                i_spectrum.append(value)
        numerical_power_spectrum.append(i_spectrum)
    return np.array(numerical_power_spectrum)


def load_gto_coefficients(args):
    return np.load(
        folder / "gto_coefficients_{n_max}_{l_max}_{r_cut}_{sigma}.npy".format(**args)
    )


def load_polynomial_coefficients(args):
    return np.load(
        folder
        / "polynomial_coefficients_{n_max}_{l_max}_{r_cut}_{sigma}.npy".format(**args)
    )


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
@pytest.mark.parametrize(
    "species, n_max, l_max, crossover, n_features",
    [
        (["H", "O"], 5, 5, True, int((5 + 1) * (5 * 2) * (5 * 2 + 1) / 2)),
        (["H", "O"], 5, 5, False, int(5 * 2 * (5 + 1) / 2 * (5 + 1))),
    ],
)
def test_number_of_features(species, n_max, l_max, crossover, n_features):
    desc = soap(
        species=species,
        r_cut=3,
        n_max=n_max,
        l_max=l_max,
        crossover=crossover,
        periodic=True,
    )
    assert_n_features(desc, n_features)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_dtype(dtype, sparse):
    assert_dtype(soap, dtype, sparse)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize(
    "centers",
    [
        "all",
        "indices_fixed",
        "indices_variable",
        "cartesian_fixed",
        "cartesian_variable",
    ],
)
def test_parallellization(n_jobs, sparse, centers):
    assert_parallellization(soap, n_jobs, sparse, centers)


@pytest.mark.parametrize("cell", ["collapsed_periodic", "collapsed_finite"])
def test_cell(cell):
    assert_cell(soap, cell)


def test_no_system_modification():
    assert_no_system_modification(soap)


def test_sparse():
    assert_sparse(soap)


@pytest.mark.parametrize("rbf", ["gto", "polynomial"])
def test_symmetries(rbf):
    # Local descriptors are not permutation symmetric.
    assert_symmetries(soap(rbf=rbf), True, True, False)


def test_periodic():
    assert_periodic(soap)


def test_centers():
    assert_centers(soap)


@pytest.mark.parametrize("rbf", ["gto", "polynomial"])
def test_basis(rbf):
    assert_basis(soap(rbf=rbf, periodic=True))


@pytest.mark.parametrize("rbf", ("gto",))
@pytest.mark.parametrize("pbc", (False, True))
@pytest.mark.parametrize("attach", (False, True))
@pytest.mark.parametrize("average", ("off", "inner", "outer"))
@pytest.mark.parametrize("crossover", (True,))
def test_derivatives_numerical(pbc, attach, average, rbf, crossover):
    descriptor_func = soap(
        r_cut=3,
        n_max=4,
        l_max=4,
        rbf=rbf,
        sparse=False,
        average=average,
        crossover=crossover,
        periodic=pbc,
        dtype="float64",
    )
    assert_derivatives(descriptor_func, "numerical", pbc, attach=attach)


@pytest.mark.parametrize("pbc, average, rbf", [(False, "off", "gto")])
@pytest.mark.parametrize("attach", (False, True))
@pytest.mark.parametrize("crossover", (True, False))
def test_derivatives_analytical(pbc, attach, average, rbf, crossover):
    descriptor_func = soap(
        r_cut=3,
        n_max=4,
        l_max=4,
        rbf=rbf,
        sparse=False,
        average=average,
        crossover=crossover,
        periodic=pbc,
        dtype="float64",
    )
    assert_derivatives(descriptor_func, "analytical", pbc, attach=attach)


@pytest.mark.parametrize("method", ("numerical", "analytical"))
def test_derivatives_include(method):
    assert_derivatives_include(soap(), method, False)


@pytest.mark.parametrize("method", ("numerical", "analytical"))
def test_derivatives_exclude(method):
    assert_derivatives_exclude(soap(), method, False)


# =============================================================================
# Tests that are specific to this descriptor.
def test_exceptions():
    """Tests different invalid parameters that should raise an
    exception.
    """
    system = get_simple_finite()

    # Invalid sigma width
    with pytest.raises(ValueError):
        SOAP(species=["H", "O"], r_cut=5, sigma=0, n_max=5, l_max=5)
    with pytest.raises(ValueError):
        SOAP(species=["H", "O"], r_cut=5, sigma=-1, n_max=5, l_max=5)

    # Invalid r_cut
    with pytest.raises(ValueError):
        SOAP(species=["H", "O"], r_cut=0.5, sigma=0.5, n_max=5, l_max=5)

    # Invalid l_max
    with pytest.raises(ValueError):
        SOAP(species=["H", "O"], r_cut=0.5, sigma=0.5, n_max=5, l_max=20, rbf="gto")
    with pytest.raises(ValueError):
        SOAP(
            species=["H", "O"],
            r_cut=0.5,
            sigma=0.5,
            n_max=5,
            l_max=21,
            rbf="polynomial",
        )

    # Invalid n_max
    with pytest.raises(ValueError):
        SOAP(species=["H", "O"], r_cut=0.5, sigma=0.5, n_max=0, l_max=21)

    # Too high radial basis set density: poly
    with pytest.raises(ValueError):
        a = SOAP(
            species=["H", "O"],
            r_cut=10,
            sigma=0.5,
            n_max=15,
            l_max=8,
            rbf="polynomial",
            periodic=False,
        )
        a.create(system)

    # Too high radial basis set density: gto
    with pytest.raises(ValueError):
        a = SOAP(
            species=["H", "O"],
            r_cut=10,
            sigma=0.5,
            n_max=20,
            l_max=8,
            rbf="gto",
            periodic=False,
        )
        a.create(system)

    def get_setup():
        return {
            "r_cut": 2,
            "sigma": 1,
            "n_max": 5,
            "l_max": 5,
            "species": ["H", "O"],
        }

    # Invalid weighting
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "poly", "c": -1, "r0": 1}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "poly", "c": 1, "r0": 0}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "poly", "c": 1, "r0": 1, "w0": -1}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "pow", "c": -1, "d": 1, "r0": 1, "m": 1}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "pow", "c": 1, "d": 1, "r0": 0, "m": 1}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "pow", "c": 1, "d": 1, "r0": 1, "w0": -1}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "exp", "c": -1, "d": 1, "r0": 1}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "exp", "c": 1, "d": 1, "r0": 0}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "exp", "c": 1, "d": 1, "r0": 1, "w0": -1}
        SOAP(**setup)
    with pytest.raises(ValueError):
        setup = get_setup()
        setup["weighting"] = {"function": "invalid", "c": 1, "d": 1, "r0": 1}
        SOAP(**setup)

    # Test that trying to get analytical derivatives with averaged output
    # raises an exception
    centers = [[0.0, 0.0, 0.0]]
    for average in ["inner", "outer"]:
        with pytest.raises(ValueError) as excinfo:
            setup = get_setup()
            setup["average"] = average
            soap = SOAP(**setup)
            soap.derivatives(system, centers=centers, method="analytical")
        assert (
            str(excinfo.value)
            == "Analytical derivatives currently not available for averaged output."
        )

    # Test that trying to get analytical derivatives with polynomial basis
    # raises an exception.
    with pytest.raises(ValueError) as excinfo:
        setup = get_setup()
        setup["rbf"] = "polynomial"
        soap = SOAP(**setup)
        soap.derivatives(system, centers=centers, method="analytical")
    assert (
        str(excinfo.value)
        == "Analytical derivatives currently not available for polynomial radial basis functions."
    )

    # Test that trying to get analytical derivatives with periodicity on
    # raises an exception
    with pytest.raises(ValueError) as excinfo:
        setup = get_setup()
        setup["periodic"] = True
        setup["rbf"] = "gto"
        soap = SOAP(**setup)
        soap.derivatives(system, centers=centers, method="analytical")
    assert (
        str(excinfo.value)
        == "Analytical derivatives currently not available for periodic systems."
    )

    # Test that trying to get analytical derivatives with weighting raises an
    # exception
    with pytest.raises(ValueError) as excinfo:
        setup = get_setup()
        setup["weighting"] = {"function": "poly", "r0": 3, "c": 1, "m": 1}
        soap = SOAP(**setup)
        soap.derivatives(system, centers=centers, method="analytical")
    assert (
        str(excinfo.value)
        == "Analytical derivatives currently not available when weighting is used."
    )


w_poly = {"function": "poly", "c": 2, "m": 3, "r0": 4}
w_pow = {"function": "pow", "threshold": 1e-3, "c": 1, "d": 1, "m": 1, "r0": 1}
w_exp = {"function": "exp", "c": 2, "d": 1, "r0": 2, "threshold": 1e-3}


@pytest.mark.parametrize(
    "weighting, expected_r_cut",
    [
        pytest.param(
            {"function": "poly", "c": 2, "m": 3, "r0": 4}, w_poly["r0"], id="poly"
        ),
        pytest.param(w_pow, w_pow["c"] * (1 / w_pow["threshold"] - 1), id="pow"),
        pytest.param(
            w_exp,
            w_exp["r0"] * np.log(w_exp["c"] / w_exp["threshold"] - w_exp["d"]),
            id="exp",
        ),
    ],
)
def test_infer_r_cut(weighting, expected_r_cut):
    """Tests that the r_cut is correctly inferred from the weighting
    function.
    """
    soap = SOAP(n_max=1, l_max=1, weighting=weighting, species=[1, 8], sparse=True)
    assert soap._r_cut == pytest.approx(expected_r_cut, rel=1e-8, abs=0)


@pytest.mark.parametrize("crossover", (False, True))
@pytest.mark.parametrize("rbf", ("gto", "polynomial"))
def test_crossover(crossover, rbf):
    """Tests that disabling/enabling crossover works as expected."""
    species = [1, 8]
    n_max = 5
    l_max = 5
    system = get_simple_finite()
    pos = [system.get_center_of_mass()]

    desc = SOAP(
        species=species,
        rbf=rbf,
        crossover=crossover,
        r_cut=3,
        n_max=n_max,
        l_max=l_max,
        periodic=False,
    )

    feat = desc.create(system, centers=pos)[0, :]
    for pair in itertools.combinations_with_replacement(species, 2):
        crossed = pair[0] != pair[1]
        if crossed:
            if not crossover:
                with pytest.raises(ValueError):
                    desc.get_location(pair)
            else:
                assert feat[desc.get_location(pair)].sum() != 0
        else:
            assert feat[desc.get_location(pair)].sum() != 0


@pytest.mark.parametrize("rbf", ["gto", "polynomial"])
def test_average_outer(rbf):
    """Tests the outer averaging (averaging done after calculating power
    spectrum).
    """
    system, centers, args = get_soap_default_setup()

    # Create the average output
    desc = SOAP(**args, rbf=rbf, average="outer")
    average = desc.create(system, centers[0:2])

    # Create individual output for both atoms
    desc = SOAP(**args, rbf=rbf, average="off")
    first = desc.create(system, [centers[0]])[0, :]
    second = desc.create(system, [centers[1]])[0, :]

    # Check that the averaging is done correctly
    assumed_average = (first + second) / 2
    assert np.allclose(average, assumed_average)


@pytest.mark.parametrize("rbf", ["gto", "polynomial"])
def test_average_inner(rbf):
    """Tests the inner averaging (averaging done before calculating power
    spectrum).
    """
    system, centers, args = globals()[f"get_soap_{rbf}_l_max_setup"]()
    # Calculate the analytical power spectrum
    soap = SOAP(**args, rbf=rbf, average="inner")
    analytical_inner = soap.create(system, centers=centers)

    # Calculate the numerical power spectrum
    coeffs = globals()[f"load_{rbf}_coefficients"](args)
    numerical_inner = get_power_spectrum(
        coeffs, crossover=args["crossover"], average="inner"
    )

    # print(f"Numerical: {numerical_inner}")
    # print(f"Analytical: {analytical_inner}")
    assert np.allclose(numerical_inner, analytical_inner, atol=1e-15, rtol=0.01)


@pytest.mark.parametrize("rbf", ["gto", "polynomial"])
def test_integration(rbf):
    """Tests that the completely analytical partial power spectrum with the
    given basis corresponds to the easier-to-code but less performant numerical
    integration done with python.
    """
    # Calculate the analytical power spectrum
    system, centers, args = globals()[f"get_soap_{rbf}_l_max_setup"]()
    soap = SOAP(**args, rbf=rbf, dtype="float64")
    analytical_power_spectrum = soap.create(system, centers=centers)

    # Fetch the precalculated numerical power spectrum
    coeffs = globals()[f"load_{rbf}_coefficients"](args)
    numerical_power_spectrum = get_power_spectrum(coeffs, crossover=args["crossover"])
    assert np.allclose(
        numerical_power_spectrum,
        analytical_power_spectrum,
        atol=1e-15,
        rtol=0.01,
    )


@pytest.mark.parametrize("rbf", ["gto", "polynomial"])
@pytest.mark.parametrize(
    "weighting",
    [
        {"function": "poly", "r0": 2, "c": 3, "m": 4},
        {"function": "pow", "r0": 2, "c": 3, "d": 4, "m": 5},
        {"function": "exp", "r0": 2, "c": 3, "d": 4},
    ],
)
def test_weighting(rbf, weighting):
    """Tests that the weighting done with C corresponds to the
    easier-to-code but less performant python version.
    """
    system, centers, args = get_soap_default_setup()

    # Calculate the analytical power spectrum
    soap = SOAP(**args, rbf=rbf, weighting=weighting)
    analytical_power_spectrum = soap.create(system, centers=centers)

    # Calculate and save the numerical power spectrum to disk
    filename = (
        folder
        / "{rbf}_coefficients_{n_max}_{l_max}_{r_cut}_{sigma}_{func}.npy".format(
            **args, rbf=rbf, func=weighting["function"]
        )
    )
    # coeffs = getattr(self, "coefficients_{}".format(rbf))(
    # system_num,
    # soap_centers_num,
    # n_max_num,
    # l_max_num,
    # r_cut_num,
    # sigma_num,
    # weighting,
    # )
    # np.save(filename, coeffs)

    # Load coefficients from disk
    coeffs = np.load(filename)
    numerical_power_spectrum = get_power_spectrum(coeffs, crossover=args["crossover"])

    # print(f"Numerical: {numerical_power_spectrum}")
    # print(f"Analytical: {analytical_power_spectrum}")
    assert np.allclose(
        numerical_power_spectrum,
        analytical_power_spectrum,
        atol=1e-15,
        rtol=0.01,
    )


def test_padding():
    """Tests that the padding used in constructing extended systems is
    sufficient.
    """
    # Fix random seed for tests
    np.random.seed(7)

    # Loop over different cell sizes
    for ncells in range(1, 6):
        ncells = int(ncells)

        # Loop over different radial cutoffs
        for r_cut in np.linspace(2, 10, 11):
            # Loop over different sigmas
            for sigma in np.linspace(0.5, 2, 4):
                # Create descriptor generators
                soap_generator = SOAP(
                    r_cut=r_cut,
                    n_max=4,
                    l_max=4,
                    sigma=sigma,
                    species=["Ni", "Ti"],
                    periodic=True,
                )

                # Define unit cell
                a = 2.993
                niti = Atoms(
                    "NiTi",
                    positions=[[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]],
                    cell=[a, a, a],
                    pbc=[1, 1, 1],
                )

                # Replicate system
                niti = niti * ncells
                a *= ncells

                # Add some noise to positions
                positions = niti.get_positions()
                noise = np.random.normal(scale=0.5, size=positions.shape)
                niti.set_positions(positions + noise)
                niti.wrap()

                # Evaluate descriptors for orthogonal unit cell
                orthogonal_soaps = soap_generator.create(niti)

                # Redefine the cubic unit cell as monoclinic with a 45-degree
                # angle, this should not affect the descriptors
                niti.set_cell([[a, 0, 0], [0, a, 0], [a, 0, a]])
                niti.wrap()

                # Evaluate descriptors for new, monoclinic unit cell
                non_orthogonal_soaps = soap_generator.create(niti)

                # Check that the relative or absolute error is small enough
                assert np.allclose(
                    orthogonal_soaps, non_orthogonal_soaps, atol=1e-8, rtol=1e-6
                )


def test_rbf_orthonormality():
    """Tests that the gto radial basis functions are orthonormal."""
    sigma = 0.15
    r_cut = 2.0
    n_max = 2
    l_max = 20
    soap = SOAP(
        species=[1],
        l_max=l_max,
        n_max=n_max,
        sigma=sigma,
        r_cut=r_cut,
        crossover=True,
        sparse=False,
    )
    alphas = np.reshape(soap._alphas, [l_max + 1, n_max])
    betas = np.reshape(soap._betas, [l_max + 1, n_max, n_max])

    nr = 10000
    n_basis = 0
    functions = np.zeros((n_max, l_max + 1, nr))

    # Form the radial basis functions
    for n in range(n_max):
        for l in range(l_max + 1):
            gto = np.zeros((nr))
            rspace = np.linspace(0, r_cut + 5, nr)
            for k in range(n_max):
                gto += (
                    betas[l, n, k] * rspace**l * np.exp(-alphas[l, k] * rspace**2)
                )
            n_basis += 1
            functions[n, l, :] = gto

    # Calculate the overlap integrals
    S = np.zeros((n_max, n_max))
    for l in range(l_max + 1):
        for i in range(n_max):
            for j in range(n_max):
                overlap = np.trapz(
                    rspace**2 * functions[i, l, :] * functions[j, l, :],
                    dx=(r_cut + 5) / nr,
                )
                S[i, j] = overlap

        # Check that the basis functions for each l are orthonormal
        diff = S - np.eye(n_max)
        assert np.allclose(diff, np.zeros((n_max, n_max)), atol=1e-3)
