import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.special import gamma


def get_basis_gto(rcut, nmax):
    """Used to calculate the alpha and beta prefactors for the gto-radial
    basis.

    Args:
        rcut(float): Radial cutoff.
        nmax(int): Number of gto radial bases.

    Returns:
        (np.ndarray, np.ndarray): The alpha and beta prefactors for all bases
        up to a fixed size of l=10.
    """
    # These are the values for where the different basis functions should decay
    # to: evenly space between 1 angstrom and rcut.
    a = np.linspace(1, rcut, nmax)
    threshold = 1e-3  # This is the fixed gaussian decay threshold

    alphas_full = np.zeros((10, nmax))
    betas_full = np.zeros((10, nmax, nmax))

    for l in range(0, 10):
        # The alphas are calculated so that the GTOs will decay to the set
        # threshold value at their respective cutoffs
        alphas = -np.log(threshold / np.power(a, l)) / a ** 2

        # Calculate the overlap matrix
        m = np.zeros((alphas.shape[0], alphas.shape[0]))
        m[:, :] = alphas
        m = m + m.transpose()
        S = 0.5 * gamma(l + 3.0 / 2.0) * m ** (-l - 3.0 / 2.0)

        # Get the beta factors that orthonormalize the set with Löwdin
        # orthonormalization
        betas = sqrtm(inv(S))

        # If the result is complex, the calculation is currently halted.
        if betas.dtype == np.complex128:
            raise ValueError(
                "Could not calculate normalization factors for the radial "
                "basis in the domain of real numbers. Lowering the number of "
                "radial basis functions (nmax) or increasing the radial "
                "cutoff (rcut) is advised."
            )

        alphas_full[l, :] = alphas
        betas_full[l, :, :] = betas

    return alphas_full, betas_full


print(get_basis_gto(5, 10))
