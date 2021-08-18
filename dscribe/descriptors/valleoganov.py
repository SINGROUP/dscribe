from dscribe.descriptors import MBTR
import math


class ValleOganov(MBTR):
    """Shortcut for implementing the fingerprint descriptor by Valle and Oganov
    for :math:`k=2` and :math:`k=3` using MBTR. Automatically uses the right
    weighting and normalizes the output, and can only be used for periodic
    systems.

    You can choose which terms to include by providing a dictionary in the
    k2 or k3 arguments. This dictionary should contain information
    under three keys: "sigma", "n" and "r_cut". See the examples below.

    For more information on the weighting and normalization used here as well
    as the other parameters and general usage of the descriptor, see the MBTR class.
    """

    def __init__(
        self,
        species,
        k2=None,
        k3=None,
        flatten=True,
        sparse=False,
    ):
        """
        Args:
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical species as low as possible is
                preferable.
            k2 (dict): Dictionary containing the setup for the k=2 term.
                Contains setup for the discretization and radial cutoff.
                For example::

                    k2 = {
                        "sigma": 0.1,
                        "n": 50,
                        "r_cut": 10
                    }

            k3 (dict): Dictionary containing the setup for the k=3 term.
                Contains setup for the discretization and radial cutoff.
                For example::

                    k3 = {
                        "sigma": 0.1,
                        "n": 50,
                        "r_cut": 10
                    }

            flatten (bool): Whether the output should be flattened to a 1D
                array. If False, a dictionary of the different tensors is
                provided, containing the values under keys: "k1", "k2", and
                "k3":
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        # Check that k2 has all the valid keys and only them
        if k2 is not None:
            for key in k2.keys():
                valid_keys = set(("sigma", "n", "r_cut"))
                if key not in valid_keys:
                    raise ValueError(
                        f"The given setup contains the following invalid key: {key}"
                    )
            for key in valid_keys:
                if key not in k2.keys():
                    raise ValueError(f"Missing value for {key}")

        # Check that k3 has all the valid keys and only them
        if k3 is not None:
            for key in k3.keys():
                valid_keys = set(("sigma", "n", "r_cut"))
                if key not in valid_keys:
                    raise ValueError(
                        f"The given setup contains the following invalid key: {key}"
                    )
            for key in valid_keys:
                if key not in k3.keys():
                    raise ValueError(f"Missing value for {key}")

        if k2 is not None:
            k2_temp = {
                "geometry": {"function": "distance"},
                "grid": {
                    "min": 0,
                    "max": k2["r_cut"],
                    "sigma": k2["sigma"],
                    "n": k2["n"],
                },
                "weighting": {"function": "inverse_square", "r_cut": k2["r_cut"]},
            }
        else:
            k2_temp = None

        if k3 is not None:
            k3_temp = {
                "geometry": {"function": "angle"},
                "grid": {"min": 0, "max": 180, "sigma": k3["sigma"], "n": k3["n"]},
                "weighting": {"function": "smooth_cutoff", "r_cut": k3["r_cut"]},
            }
        else:
            k3_temp = None

        super().__init__(
            species=species,
            periodic=True,
            k1=None,
            k2=k2_temp,
            k3=k3_temp,
            flatten=flatten,
            sparse=sparse,
            normalization="valle_oganov",
            normalize_gaussians=True,
        )
