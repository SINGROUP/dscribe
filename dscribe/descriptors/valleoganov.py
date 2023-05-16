from dscribe.descriptors import MBTR


class ValleOganov(MBTR):
    """Shortcut for implementing the fingerprint descriptor by Valle and Oganov
    for using MBTR. Automatically uses the right weighting and normalizes the
    output, and can only be used for periodic systems.

    For more information on the weighting and normalization used here as well
    as the other parameters and general usage of the descriptor, see the MBTR class.
    """

    def __init__(
        self,
        species,
        function,
        n,
        sigma,
        r_cut,
        sparse=False,
        dtype="float64",
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

            function (str): The geometry function. The order :math:`k`
                tells how many atoms are involved in the calculation and thus
                also heavily influence the computational time.

                The following geometry functions are available:

                * :math:`k=2`
                    * ``"distance"``: Pairwise distance in angstroms.
                * :math:`k=3`
                    * ``"angle"``: Angle in degrees.

            n (int): Number of discretization points.
            sigma (float): Standard deviation of the gaussian broadening
            r_cut (float): Radial cutoff.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.

            dtype (str): The data type of the output. Valid options are:

                    * ``"float32"``: Single precision floating point numbers.
                    * ``"float64"``: Double precision floating point numbers.
        """
        if function == "distance":
            geometry = {"function": "distance"}
            grid = {
                "min": 0,
                "max": r_cut,
                "sigma": sigma,
                "n": n,
            }
            weighting = {"function": "inverse_square", "r_cut": r_cut}
        elif function == "angle":
            geometry = {"function": "angle"}
            grid = {"min": 0, "max": 180, "sigma": sigma, "n": n}
            weighting = {"function": "smooth_cutoff", "r_cut": r_cut}
        else:
            raise ValueError("Invalid function. Use 'distance' or 'angle'.")

        super().__init__(
            species=species,
            periodic=True,
            grid=grid,
            geometry=geometry,
            weighting=weighting,
            sparse=sparse,
            normalization="valle_oganov",
            normalize_gaussians=True,
            dtype=dtype,
        )
