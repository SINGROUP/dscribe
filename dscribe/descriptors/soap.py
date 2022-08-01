# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import time
import numpy as np
from warnings import warn

from scipy.special import gamma
from scipy.linalg import sqrtm, inv

from ase import Atoms
import ase.geometry.cell
import ase.data

import sparse as sp

from dscribe.descriptors import Descriptor
from dscribe.core import System
from dscribe.utils.dimensionality import is1d
import dscribe.ext


class SOAP(Descriptor):
    """Class for generating a partial power spectrum from Smooth Overlap of
    Atomic Orbitals (SOAP). This implementation uses real (tesseral) spherical
    harmonics as the angular basis set and provides two orthonormalized
    alternatives for the radial basis functions: spherical primitive gaussian
    type orbitals ("gto") or the polynomial basis set ("polynomial").

    For reference, see:

    "On representing chemical environments, Albert P. Bartók, Risi Kondor, and
    Gábor Csányi, Phys. Rev. B 87, 184115, (2013),
    https://doi.org/10.1103/PhysRevB.87.184115

    "Comparing molecules and solids across structural and alchemical space",
    Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti, Phys.
    Chem. Chem. Phys. 18, 13754 (2016), https://doi.org/10.1039/c6cp00415f

    "Machine learning hydrogen adsorption on nanoclusters through structural
    descriptors", Marc O. J. Jäger, Eiaki V. Morooka, Filippo Federici Canova,
    Lauri Himanen & Adam S. Foster, npj Comput. Mater., 4, 37 (2018),
    https://doi.org/10.1038/s41524-018-0096-5
    """

    def __init__(
        self,
        r_cut=None,
        n_max=None,
        l_max=None,
        sigma=1.0,
        rbf="gto",
        weighting=None,
        crossover=True,
        average="off",
        species=None,
        periodic=False,
        sparse=False,
        dtype="float64",
        # For backwards compatibility with < v1.2.2
        rcut=None,
        nmax=None,
        lmax=None,
    ):
        """
        Args:
            r_cut (float): A cutoff for local region in angstroms. Should be
                bigger than 1 angstrom for the gto-basis.
            n_max (int): The number of radial basis functions.
            l_max (int): The maximum degree of spherical harmonics.
            sigma (float): The standard deviation of the gaussians used to expand the
                atomic density.
            rbf (str): The radial basis functions to use. The available options are:

                * ``"gto"``: Spherical gaussian type orbitals defined as :math:`g_{nl}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'l} r^l e^{-\\alpha_{n'l}r^2}`
                * ``"polynomial"``: Polynomial basis defined as :math:`g_{n}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'} (r-r_\mathrm{cut})^{n'+2}`

            weighting (dict): Contains the options which control the
                weighting of the atomic density. Leave unspecified if
                you do not wish to apply any weighting. The dictionary may
                contain the following entries:

                * ``"function"``: The weighting function to use. The
                  following are currently supported:

                    * ``"poly"``: :math:`w(r) = \left\{ \\begin{array}{ll} c(1 + 2 (\\frac{r}{r_0})^{3} -3 (\\frac{r}{r_0})^{2}))^{m}, \\ \\text{for}\\ r \\leq r_0\\\\ 0, \\ \\text{for}\\ r > r_0 \end{array}\\right.`

                      This function goes exactly to zero at :math:`r=r_0`. If
                      you do not explicitly provide ``r_cut`` in the
                      constructor, ``r_cut`` is automatically set to ``r0``.
                      You can provide the parameters ``c``, ``m`` and ``r0`` as
                      additional dictionary items.

                      For reference see:
                          "Caro, M. (2019). Optimizing many-body atomic
                          descriptors for enhanced computational performance of
                          machine learning based interatomic potentials.  Phys.
                          Rev. B, 100, 024112."

                    * ``"pow"``: :math:`w(r) = \\frac{c}{d + (\\frac{r}{r_0})^{m}}`

                      If you do not explicitly provide ``r_cut`` in the
                      constructor, ``r_cut`` will be set as the value at which
                      this function decays to the value given by the
                      ``threshold`` entry in the weighting dictionary (defaults
                      to 1e-2), You can provide the parameters ``c``, ``d``,
                      ``m``, ``r0`` and ``threshold`` as additional dictionary
                      items.

                      For reference see:
                          "Willatt, M., Musil, F., & Ceriotti, M. (2018).
                          Feature optimization for atomistic machine learning
                          yields a data-driven construction of the periodic
                          table of the elements.  Phys. Chem. Chem. Phys., 20,
                          29661-29668.
                          "

                    * ``"exp"``: :math:`w(r) = \\frac{c}{d + e^{-r/r_0}}`

                      If you do not explicitly provide ``r_cut`` in the
                      constructor, ``r_cut`` will be set as the value at which
                      this function decays to the value given by the
                      ``threshold`` entry in the weighting dictionary (defaults
                      to 1e-2), You can provide the parameters ``c``, ``d``,
                      ``r0`` and ``threshold`` as additional dictionary items.

                * ``"w0"``: Optional weight for atoms that are directly on top
                  of a requested center. Setting this value to zero essentially
                  hides the central atoms from the output. If a weighting
                  function is also specified, this constant will override it
                  for the central atoms.

            crossover (bool): Determines if crossover of atomic types should
                be included in the power spectrum. If enabled, the power
                spectrum is calculated over all unique species combinations Z
                and Z'. If disabled, the power spectrum does not contain
                cross-species information and is only run over each unique
                species Z. Turned on by default to correspond to the original
                definition
            average (str): The averaging mode over the centers of interest.
                Valid options are:

                    * ``"off"``: No averaging.
                    * ``"inner"``: Averaging over sites before summing up the magnetic quantum numbers: :math:`p_{nn'l}^{Z_1,Z_2} \sim \sum_m (\\frac{1}{n} \sum_i c_{nlm}^{i, Z_1})^{*} (\\frac{1}{n} \sum_i c_{n'lm}^{i, Z_2})`
                    * ``"outer"``: Averaging over the power spectrum of different sites: :math:`p_{nn'l}^{Z_1,Z_2} \sim \\frac{1}{n} \sum_i \sum_m (c_{nlm}^{i, Z_1})^{*} (c_{n'lm}^{i, Z_2})`
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical species as low as possible is
                preferable.
            periodic (bool): Set to True if you want the descriptor output to
                respect the periodicity of the atomic systems (see the
                pbc-parameter in the constructor of ase.Atoms).
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
            dtype (str): The data type of the output. Valid options are:

                    * ``"float32"``: Single precision floating point numbers.
                    * ``"float64"``: Double precision floating point numbers.

        """
        var_dict = {}
        for var_new in ["r_cut", "n_max", "l_max"]:
            loc = locals()
            var_old = "".join(var_new.split("_"))
            if loc.get(var_old) is not None:
                var_dict[var_new] = loc[var_old]
                if loc.get(var_new) is not None:
                    raise ValueError(
                        "Please provide only either {} or {}.".format(var_new, var_old)
                    )
            else:
                var_dict[var_new] = loc[var_new]
        r_cut = var_dict["r_cut"]
        n_max = var_dict["n_max"]
        l_max = var_dict["l_max"]

        supported_dtype = set(("float32", "float64"))
        if dtype not in supported_dtype:
            raise ValueError(
                "Invalid output data type '{}' given. Please use "
                "one of the following: {}".format(dtype, supported_dtype)
            )
        super().__init__(periodic=periodic, flatten=True, sparse=sparse, dtype=dtype)

        # Setup the involved chemical species
        self.species = species

        # Test that general settings are valid
        if sigma <= 0:
            raise ValueError(
                "Only positive gaussian width parameters 'sigma' are allowed."
            )
        self._eta = 1 / (2 * sigma**2)
        self._sigma = sigma

        supported_rbf = {"gto", "polynomial"}
        if rbf not in supported_rbf:
            raise ValueError(
                "Invalid radial basis function of type '{}' given. Please use "
                "one of the following: {}".format(rbf, supported_rbf)
            )
        if n_max < 1:
            raise ValueError(
                "Must have at least one radial basis function." "n_max={}".format(n_max)
            )
        supported_average = set(("off", "inner", "outer"))
        if average not in supported_average:
            raise ValueError(
                "Invalid average mode '{}' given. Please use "
                "one of the following: {}".format(average, supported_average)
            )

        if not (weighting or r_cut):
            raise ValueError("Either weighting or r_cut need to be defined")
        if weighting:
            w0 = weighting.get("w0")
            if w0 is not None:
                if w0 < 0:
                    raise ValueError("Define w0 > 0 in weighting.")
                weighting["w0"] = float(w0)
            func = weighting.get("function")
            if func is not None:
                weighting_functions = {"poly", "pow", "exp"}
                if func not in weighting_functions:
                    raise ValueError(
                        "Weighting function not implemented. Please choose "
                        "among one of the following {}".format(weighting_functions)
                    )
                r0 = weighting.get("r0")
                if r0 is None or r0 <= 0:
                    raise ValueError("Define r0 > 0 in weighting.")
                weighting["r0"] = float(r0)
                c = weighting.get("c")
                if c is None or c < 0:
                    raise ValueError("Define c >= 0 in weighting.")
                weighting["c"] = float(c)
                if func == "poly":
                    m = weighting.get("m")
                    if m is None or m < 0:
                        raise ValueError("Define m >= 0 in weighting.")
                    weighting["m"] = float(m)
                elif func == "pow":
                    d = weighting.get("d")
                    if d is None or d < 0:
                        raise ValueError("Define d >= 0 in weighting.")
                    weighting["d"] = float(d)
                    m = weighting.get("m")
                    if m is None or m < 0:
                        raise ValueError("Define m >= 0 in weighting.")
                    weighting["m"] = float(m)
                    weighting["threshold"] = float(weighting.get("threshold", 1e-2))
                elif func == "exp":
                    d = weighting.get("d")
                    if d < 0:
                        raise ValueError("Define d >= 0 in weighting.")
                    weighting["d"] = float(d)
                    weighting["threshold"] = float(weighting.get("threshold", 1e-2))
        else:
            weighting = {}
        if not r_cut:
            r_cut = self._infer_r_cut(weighting)

        # Test that radial basis set specific settings are valid
        if rbf == "gto":
            if r_cut <= 1:
                raise ValueError(
                    "When using the gaussian radial basis set (gto), the radial "
                    "cutoff should be bigger than 1 angstrom."
                )
            # Precalculate the alpha and beta constants for the GTO basis
            self._alphas, self._betas = self.get_basis_gto(r_cut, n_max, l_max)

        # Test l_max
        if l_max < 0:
            raise ValueError("l_max cannot be negative. l_max={}".format(l_max))
        elif l_max > 20:
            raise ValueError(
                "The maximum available l_max for SOAP is currently 20, you have"
                " requested l_max={}".format(l_max)
            )

        self._r_cut = float(r_cut)
        self._weighting = weighting
        self._n_max = n_max
        self._l_max = l_max
        self._rbf = rbf
        self.average = average
        self.crossover = crossover

    def prepare_centers(self, system, cutoff_padding, positions=None):
        """Validates and prepares the centers for the C++ extension."""
        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        self.check_atomic_numbers(system.get_atomic_numbers())

        # Check if periodic is valid
        if self.periodic:
            cell = system.get_cell()
            if np.cross(cell[0], cell[1]).dot(cell[2]) == 0:
                raise ValueError("System doesn't have cell to justify periodicity.")

        # Setup the local positions
        if positions is None:
            list_positions = system.get_positions()
            indices = np.arange(len(system))
        else:
            # Check validity of position definitions and create final cartesian
            # position list
            error = ValueError(
                "The argument 'positions' should contain a non-empty "
                "one-dimensional list of"
                " atomic indices or a two-dimensional "
                "list of cartesian coordinates with x, y and z components."
            )
            if (
                not isinstance(positions, (list, tuple, np.ndarray))
                or len(positions) == 0
            ):
                raise error
            list_positions = []
            indices = np.full(len(positions), -1, dtype=np.int)
            for idx, i in enumerate(positions):
                if np.issubdtype(type(i), np.integer):
                    list_positions.append(system.get_positions()[i])
                    indices[idx] = i
                elif isinstance(i, (list, tuple, np.ndarray)):
                    if len(i) != 3:
                        raise error
                    list_positions.append(i)
                else:
                    raise error

        return np.asarray(list_positions), indices

    def get_cutoff_padding(self):
        """The radial cutoff is extended by adding a padding that depends on
        the used used sigma value. The padding is chosen so that the gaussians
        decay to the specified threshold value at the cutoff distance.
        """
        threshold = 0.001
        cutoff_padding = self._sigma * np.sqrt(-2 * np.log(threshold))
        return cutoff_padding

    def _infer_r_cut(self, weighting):
        """Used to determine an appropriate r_cut based on where the given
        weighting function setup.
        """
        if weighting["function"] == "pow":
            t = weighting["threshold"]
            m = weighting["m"]
            c = weighting["c"]
            d = weighting["d"]
            r0 = weighting["r0"]
            r_cut = r0 * (c / t - d) ** (1 / m)
            return r_cut
        elif weighting["function"] == "poly":
            r0 = weighting["r0"]
            return r0
        elif weighting["function"] == "exp":
            t = weighting["threshold"]
            c = weighting["c"]
            d = weighting["d"]
            r0 = weighting["r0"]
            r_cut = r0 * np.log(c / t - d)
            return r_cut
        else:
            return None

    def init_descriptor_array(self, n_centers, n_features):
        """Return a zero-initialized numpy array for the descriptor."""
        if self.average == "inner" or self.average == "outer":
            c = np.zeros((1, n_features), dtype=np.float64)
        else:
            c = np.zeros((n_centers, n_features), dtype=np.float64)
        return c

    def init_derivatives_array(self, n_centers, n_indices, n_features):
        """Return a zero-initialized numpy array for the derivatives."""
        if self.average == "inner" or self.average == "outer":
            return np.zeros((1, n_indices, 3, n_features), dtype=np.float64)
        else:
            return np.zeros((n_centers, n_indices, 3, n_features), dtype=np.float64)

    def init_internal_dev_array(self, n_centers, n_atoms, n_types, n, l_max):
        d = np.zeros(
            (n_atoms, n_centers, n_types, n, (l_max + 1) * (l_max + 1)),
            dtype=np.float64,
        )
        return d

    def init_internal_array(self, n_centers, n_types, n, l_max):
        d = np.zeros(
            (n_centers, n_types, n, (l_max + 1) * (l_max + 1)), dtype=np.float64
        )
        return d

    def create(
        self, system, positions=None, n_jobs=1, only_physical_cores=False, verbose=False
    ):
        """Return the SOAP output for the given systems and given positions.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            positions (list): Positions where to calculate SOAP. Can be
                provided as cartesian positions or atomic indices. If no
                positions are defined, the SOAP output will be created for all
                atoms in the system. When calculating SOAP for multiple
                systems, provide the positions as a list for each system.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the used cpus
                will be calculated with, n_cpus + n_jobs, where n_cpus is the
                amount of CPUs as reported by the OS. With only_physical_cores
                you can control which types of CPUs are counted in n_cpus.
            only_physical_cores (bool): If a negative n_jobs is given,
                determines which types of CPUs are used in calculating the
                number of jobs. If set to False (default), also virtual CPUs
                are counted.  If set to True, only physical CPUs are counted.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            np.ndarray | sparse.COO: The SOAP output for the given systems and
            positions. The return type depends on the 'sparse'-attribute. The
            first dimension is determined by the amount of positions and
            systems and the second dimension is determined by the
            get_number_of_features()-function. When multiple systems are
            provided the results are ordered by the input order of systems and
            their positions.
        """
        # Validate input / combine input arguments
        if isinstance(system, Atoms):
            system = [system]
            positions = [positions]
        n_samples = len(system)
        if positions is None:
            inp = [(i_sys,) for i_sys in system]
        else:
            n_pos = len(positions)
            if n_pos != n_samples:
                raise ValueError(
                    "The given number of positions does not match the given "
                    "number of systems."
                )
            inp = list(zip(system, positions))

        # Determine if the outputs have a fixed size
        n_features = self.get_number_of_features()
        static_size = None
        if self.average == "outer" or self.average == "inner":
            static_size = [n_features]
        else:
            if positions is None:
                n_centers = len(inp[0][0])
            else:
                first_sample, first_pos = inp[0]
                if first_pos is not None:
                    n_centers = len(first_pos)
                else:
                    n_centers = len(first_sample)

            def is_static():
                for i_job in inp:
                    if positions is None:
                        if len(i_job[0]) != n_centers:
                            return False
                    else:
                        if i_job[1] is not None:
                            if len(i_job[1]) != n_centers:
                                return False
                        else:
                            if len(i_job[0]) != n_centers:
                                return False
                return True

            if is_static():
                static_size = [n_centers, n_features]

        # Create in parallel
        output = self.create_parallel(
            inp,
            self.create_single,
            n_jobs,
            static_size,
            only_physical_cores,
            verbose=verbose,
        )

        return output

    def create_single(self, system, positions=None):
        """Return the SOAP output for the given system and given positions.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            positions (list): Cartesian positions or atomic indices. If
                specified, the SOAP spectrum will be created for these points.
                If no positions are defined, the SOAP output will be created
                for all atoms in the system.

        Returns:
            np.ndarray | sparse.COO: The SOAP output for the
            given system and positions. The return type depends on the
            'sparse'-attribute. The first dimension is given by the number of
            positions and the second dimension is determined by the
            get_number_of_features()-function.
        """
        cutoff_padding = self.get_cutoff_padding()
        centers, center_indices = self.prepare_centers(
            system, cutoff_padding, positions
        )
        n_centers = centers.shape[0]
        n_species = self._atomic_numbers.shape[0]
        pos = system.get_positions()
        Z = system.get_atomic_numbers()
        n_features = self.get_number_of_features()
        n_atoms = Z.shape[0]
        soap_mat = self.init_descriptor_array(n_centers, n_features)

        # Determine the function to call based on rbf
        if self._rbf == "gto":

            # Orthonormalized RBF coefficients
            alphas = self._alphas.flatten()
            betas = self._betas.flatten()

            # Determine shape
            n_features = self.get_number_of_features()
            soap_mat = self.init_descriptor_array(n_centers, n_features)

            # Calculate with extension
            soap_gto = dscribe.ext.SOAPGTO(
                self._r_cut,
                self._n_max,
                self._l_max,
                self._eta,
                self._weighting,
                self.crossover,
                self.average,
                cutoff_padding,
                alphas,
                betas,
                self._atomic_numbers,
                self.periodic,
            )

            # Calculate analytically with extension
            soap_gto.create(
                soap_mat,
                pos,
                Z,
                ase.geometry.cell.complete_cell(system.get_cell()),
                np.asarray(system.get_pbc(), dtype=bool),
                centers,
            )
        elif self._rbf == "polynomial":
            # Get the discretized and orthogonalized polynomial radial basis
            # function values
            rx, gss = self.get_basis_poly(self._r_cut, self._n_max)
            gss = gss.flatten()

            # Calculate with extension
            soap_poly = dscribe.ext.SOAPPolynomial(
                self._r_cut,
                self._n_max,
                self._l_max,
                self._eta,
                self._weighting,
                self.crossover,
                self.average,
                cutoff_padding,
                rx,
                gss,
                self._atomic_numbers,
                self.periodic,
            )
            soap_poly.create(
                soap_mat,
                pos,
                Z,
                ase.geometry.cell.complete_cell(system.get_cell()),
                np.asarray(system.get_pbc(), dtype=bool),
                centers,
            )

        # Averaged output is a global descriptor, and thus the first dimension
        # is squeezed out to keep the output size consistent with the size of
        # other global descriptors.
        if self.average != "off":
            soap_mat = np.squeeze(soap_mat, axis=0)

        # Convert to the final output precision.
        if self.dtype != "float64":
            soap_mat = soap_mat.astype(self.dtype)

        # Make into a sparse array if requested
        if self._sparse:
            soap_mat = sp.COO.from_numpy(soap_mat)

        return soap_mat

    def derivatives(
        self,
        system,
        positions=None,
        include=None,
        exclude=None,
        method="auto",
        return_descriptor=True,
        attach=False,
        n_jobs=1,
        only_physical_cores=False,
        verbose=False,
    ):
        """Return the descriptor derivatives for the given systems and given positions.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            positions (list): Positions where to calculate the descriptor. Can be
                provided as cartesian positions or atomic indices. Also see the
                "attach"-argument that controls the interperation of locations
                given as atomic indices. If no positions are defined, the
                descriptor output will be created for all atoms in the system.
                When calculating descriptor for multiple systems, provide the
                positions as a list for each system.
            include (list): Indices of atoms to compute the derivatives on.
                When calculating descriptor for multiple systems, provide
                either a one-dimensional list that if applied to all systems or
                a two-dimensional list of indices. Cannot be provided together
                with 'exclude'.
            exclude (list): Indices of atoms not to compute the derivatives on.
                When calculating descriptor for multiple systems, provide
                either a one-dimensional list that if applied to all systems or
                a two-dimensional list of indices. Cannot be provided together
                with 'include'.
            method (str): The method for calculating the derivatives. Provide
                either 'numerical', 'analytical' or 'auto'. If using 'auto',
                the most efficient available method is automatically chosen.
            attach (bool): Controls the behaviour of positions defined as
                atomic indices. If True, the positions tied to an atomic index will
                move together with the atoms with respect to which the derivatives
                are calculated against. If False, positions defined as atomic
                indices will be converted into cartesian locations that are
                completely independent of the atom location during derivative
                calculation.
            return_descriptor (bool): Whether to also calculate the descriptor
                in the same function call. Notice that it typically is faster
                to calculate both in one go.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the number of jobs
                will be calculated with, n_cpus + n_jobs, where n_cpus is the
                amount of CPUs as reported by the OS. With only_physical_cores
                you can control which types of CPUs are counted in n_cpus.
            only_physical_cores (bool): If a negative n_jobs is given,
                determines which types of CPUs are used in calculating the
                number of jobs. If set to False (default), also virtual CPUs
                are counted.  If set to True, only physical CPUs are counted.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            If return_descriptor is True, returns a tuple, where the first item
            is the derivative array and the second is the descriptor array.
            Otherwise only returns the derivatives array. The derivatives array
            is a either a 4D or 5D array, depending on whether you have
            provided a single or multiple systems. If the output shape for each
            system is the same, a single monolithic numpy array is returned.
            For variable sized output (e.g. differently sized systems,
            different number of centers or different number of included atoms),
            a regular python list is returned. The dimensions are:
            [(n_systems,) n_positions, n_atoms, 3, n_features]. The first
            dimension goes over the different systems in case multiple were
            given. The second dimension goes over the descriptor centers in
            the same order as they were given in the argument. The third
            dimension goes over the included atoms. The order is same as the
            order of atoms in the given system. The fourth dimension goes over
            the cartesian components, x, y and z. The fifth dimension goes over
            the features in the default order.
        """
        methods = {"numerical", "analytical", "auto"}
        if method not in methods:
            raise ValueError(
                "Invalid method specified. Please choose from: {}".format(methods)
            )
        if self.average != "off" and method == "analytical":
            raise ValueError(
                "Analytical derivatives not currently available for averaged output."
            )
        if self.periodic:
            raise ValueError(
                "Derivatives are currently not available for periodic systems."
            )
        if self._rbf == "polynomial" and method == "analytical":
            raise ValueError(
                "Analytical derivatives not currently available for polynomial "
                "radial basis set."
            )
        if attach and method == "analytical":
            raise ValueError(
                "Analytical derivatives not currently available when attach=True."
            )

        # Determine the appropriate method if not given explicitly.
        if method == "auto":
            if self._rbf == "polynomial" or self.average != "off" or attach:
                method = "numerical"
            else:
                method = "analytical"

        # If single system given, skip the parallelization
        if isinstance(system, (Atoms, System)):
            n_atoms = len(system)
            indices = self._get_indices(n_atoms, include, exclude)
            return self.derivatives_single(
                system,
                positions,
                indices,
                method=method,
                attach=attach,
                return_descriptor=return_descriptor,
            )

        # Check input validity
        n_samples = len(system)
        if positions is None:
            positions = [None] * n_samples
        if include is None:
            include = [None] * n_samples
        elif is1d(include, np.integer):
            include = [include] * n_samples
        if exclude is None:
            exclude = [None] * n_samples
        elif is1d(exclude, np.integer):
            exclude = [exclude] * n_samples
        n_pos = len(positions)
        if n_pos != n_samples:
            raise ValueError(
                "The given number of positions does not match the given "
                "number of systems."
            )
        n_inc = len(include)
        if n_inc != n_samples:
            raise ValueError(
                "The given number of includes does not match the given "
                "number of systems."
            )
        n_exc = len(exclude)
        if n_exc != n_samples:
            raise ValueError(
                "The given number of excludes does not match the given "
                "number of systems."
            )

        # Determine the atom indices that are displaced
        indices = []
        for sys, inc, exc in zip(system, include, exclude):
            n_atoms = len(sys)
            indices.append(self._get_indices(n_atoms, inc, exc))

        # Combine input arguments
        inp = list(
            zip(
                system,
                positions,
                indices,
                [method] * n_samples,
                [attach] * n_samples,
                [return_descriptor] * n_samples,
            )
        )

        # For the descriptor, the output size for each job depends on the exact arguments.
        # Here we precalculate the size for each job to preallocate memory and
        # make the process faster.
        n_features = self.get_number_of_features()

        def get_shapes(job):
            centers = job[1]
            if centers is None:
                n_positions = len(job[0])
            else:
                n_positions = 1 if self.average != "off" else len(centers)
            n_indices = len(job[2])
            return (n_positions, n_indices, 3, n_features), (n_positions, n_features)

        derivatives_shape, descriptor_shape = get_shapes(inp[0])

        def is_variable(inp):
            for job in inp[1:]:
                i_der_shape, i_desc_shape = get_shapes(job)
                if i_der_shape != derivatives_shape or i_desc_shape != descriptor_shape:
                    return True
            return False

        if is_variable(inp):
            derivatives_shape = None
            descriptor_shape = None

        # Create in parallel
        output = self.derivatives_parallel(
            inp,
            self.derivatives_single,
            n_jobs,
            derivatives_shape,
            descriptor_shape,
            return_descriptor,
            only_physical_cores,
            verbose=verbose,
        )

        return output

    def derivatives_single(
        self,
        system,
        positions,
        indices,
        method="numerical",
        attach=False,
        return_descriptor=True,
    ):
        """Return the SOAP output for the given system and given positions.

        Args:
            system (:class:`ase.Atoms`): Atomic structure.
            positions (list): Positions where to calculate SOAP. Can be
                provided as cartesian positions or atomic indices. If no
                positions are defined, the SOAP output will be created for all
                atoms in the system. When calculating SOAP for multiple
                systems, provide the positions as a list for each system.
            indices (list): Indices of atoms for which the derivatives will be
                computed for.
            method (str): 'numerical' or 'analytical' derivatives. Numerical
                derivatives are implemented with central finite difference. If
                not specified, analytical derivatives are used when available.
            attach (bool): Controls the behaviour of positions defined as
                atomic indices. If True, the positions tied to an atomic index will
                move together with the atoms with respect to which the derivatives
                are calculated against. If False, positions defined as atomic
                indices will be converted into cartesian locations that are
                completely independent of the atom location during derivative
                calculation.
            return_descriptor (bool): Whether to also calculate the descriptor
                in the same function call. This is true by default as it
                typically is faster to calculate both in one go.

        Returns:
            If return_descriptor is True, returns a tuple, where the first item
            is the derivative array and the second is the descriptor array.
            Otherwise only returns the derivatives array. The derivatives array
            is a 4D numpy array. The dimensions are: [n_positions, n_atoms, 3,
            n_features]. The first dimension goes over the SOAP centers in the
            same order as they were given in the argument. The second dimension
            goes over the included atoms. The order is same as the order of
            atoms in the given system. The third dimension goes over the
            cartesian components, x, y and z. The last dimension goes over the
            features in the default order.
        """
        cutoff_padding = self.get_cutoff_padding()
        centers, center_indices = self.prepare_centers(
            system, cutoff_padding, positions
        )
        pos = system.get_positions()
        Z = system.get_atomic_numbers()
        sorted_species = self._atomic_numbers
        n_species = len(sorted_species)
        n_centers = centers.shape[0]
        n_indices = len(indices)
        n_atoms = len(system)

        n_features = self.get_number_of_features()
        if return_descriptor:
            c = self.init_descriptor_array(n_centers, n_features)
        else:
            c = np.empty(0)
        d = self.init_derivatives_array(n_centers, n_indices, n_features)

        if self._rbf == "gto":
            alphas = self._alphas.flatten()
            betas = self._betas.flatten()
            soap_gto = dscribe.ext.SOAPGTO(
                self._r_cut,
                self._n_max,
                self._l_max,
                self._eta,
                self._weighting,
                self.crossover,
                self.average,
                cutoff_padding,
                alphas,
                betas,
                self._atomic_numbers,
                self.periodic,
            )

            # Calculate numerically with extension
            if method == "numerical":
                soap_gto.derivatives_numerical(
                    d,
                    c,
                    pos,
                    Z,
                    ase.geometry.cell.complete_cell(system.get_cell()),
                    np.asarray(system.get_pbc(), dtype=bool),
                    centers,
                    center_indices,
                    indices,
                    attach,
                    return_descriptor,
                )
            # Calculate analytically with extension
            elif method == "analytical":
                # These arrays are only used internally by the C++ code.
                # Allocating them here with python is much faster than
                # allocating similarly sized arrays within C++. It seems
                # that numpy does some kind of lazy allocation that is
                # highly efficient for zero-initialized arrays. Similar
                # performace could not be achieved even with calloc.
                xd = self.init_internal_dev_array(
                    n_centers, n_atoms, n_species, self._n_max, self._l_max
                )
                yd = self.init_internal_dev_array(
                    n_centers, n_atoms, n_species, self._n_max, self._l_max
                )
                zd = self.init_internal_dev_array(
                    n_centers, n_atoms, n_species, self._n_max, self._l_max
                )

                soap_gto.derivatives_analytical(
                    d,
                    c,
                    xd,
                    yd,
                    zd,
                    pos,
                    Z,
                    ase.geometry.cell.complete_cell(system.get_cell()),
                    np.asarray(system.get_pbc(), dtype=bool),
                    centers,
                    indices,
                    return_descriptor,
                )

        elif self._rbf == "polynomial":
            rx, gss = self.get_basis_poly(self._r_cut, self._n_max)
            gss = gss.flatten()

            # Calculate numerically with extension
            if method == "numerical":
                soap_poly = dscribe.ext.SOAPPolynomial(
                    self._r_cut,
                    self._n_max,
                    self._l_max,
                    self._eta,
                    self._weighting,
                    self.crossover,
                    self.average,
                    cutoff_padding,
                    rx,
                    gss,
                    self._atomic_numbers,
                    self.periodic,
                )
                soap_poly.derivatives_numerical(
                    d,
                    c,
                    pos,
                    Z,
                    ase.geometry.cell.complete_cell(system.get_cell()),
                    np.asarray(system.get_pbc(), dtype=bool),
                    centers,
                    center_indices,
                    indices,
                    attach,
                    return_descriptor,
                )

        # Convert to the final output precision.
        if self.dtype == "float32":
            d = d.astype(self.dtype)
            c = c.astype(self.dtype)

        # Convert to sparse here. Currently everything up to this point is
        # calculated with dense matrices. This imposes some memory limitations.
        if self.sparse:
            d = sp.COO.from_numpy(d)
            c = sp.COO.from_numpy(c)

        if return_descriptor:
            return (d, c)
        return d

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        """Used to check the validity of given atomic numbers and to initialize
        the C-memory layout for them.

        Args:
            value(iterable): Chemical species either as a list of atomic
                numbers or list of chemical symbols.
        """
        # The species are stored as atomic numbers for internal use.
        self._set_species(value)

        # Setup mappings between atom indices and types
        self.atomic_number_to_index = {}
        self.index_to_atomic_number = {}
        for i_atom, atomic_number in enumerate(self._atomic_numbers):
            self.atomic_number_to_index[atomic_number] = i_atom
            self.index_to_atomic_number[i_atom] = atomic_number
        self.n_elements = len(self._atomic_numbers)

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_elem = len(self._atomic_numbers)
        if self.crossover:
            n_elem_radial = n_elem * self._n_max
            return int((n_elem_radial) * (n_elem_radial + 1) / 2 * (self._l_max + 1))
        else:
            return int(n_elem * self._n_max * (self._n_max + 1) / 2 * (self._l_max + 1))

    def get_location(self, species):
        """Can be used to query the location of a species combination in the
        the flattened output.

        Args:
            species(tuple): A tuple containing a pair of species as chemical
                symbols or atomic numbers. The tuple can be for example ("H", "O").

        Returns:
            slice: slice containing the location of the specified species
            combination. The location is given as a python slice-object, that
            can be directly used to target ranges in the output.

        Raises:
            ValueError: If the requested species combination is not in the
                output or if invalid species defined.
        """
        # Check that the corresponding part is calculated
        if len(species) != 2:
            raise ValueError("Please use a pair of atomic numbers or chemical symbols.")

        # Change chemical elements into atomic numbers
        numbers = []
        for specie in species:
            if isinstance(specie, str):
                try:
                    specie = ase.data.atomic_numbers[specie]
                except KeyError:
                    raise ValueError("Invalid chemical species: {}".format(specie))
            numbers.append(specie)

        # See if species defined
        for number in numbers:
            if number not in self._atomic_number_set:
                raise ValueError(
                    "Atomic number {} was not specified in the species.".format(number)
                )

        # Change into internal indexing
        numbers = [self.atomic_number_to_index[x] for x in numbers]
        n_elem = self.n_elements

        if numbers[0] > numbers[1]:
            numbers = list(reversed(numbers))
        i = numbers[0]
        j = numbers[1]
        n_elem_feat_symm = self._n_max * (self._n_max + 1) / 2 * (self._l_max + 1)

        if self.crossover:
            n_elem_feat_unsymm = self._n_max * self._n_max * (self._l_max + 1)
            n_elem_feat = n_elem_feat_symm if i == j else n_elem_feat_unsymm

            # The diagonal terms are symmetric and off-diagonal terms are
            # unsymmetric
            m_symm = i + int(j > i)
            m_unsymm = j + i * n_elem - i * (i + 1) / 2 - m_symm

            start = int(m_symm * n_elem_feat_symm + m_unsymm * n_elem_feat_unsymm)
            end = int(start + n_elem_feat)
        else:
            if i != j:
                raise ValueError(
                    "Crossover is set to False. No cross-species output " "available"
                )
            start = int(i * n_elem_feat_symm)
            end = int(start + n_elem_feat_symm)

        return slice(start, end)

    def get_basis_gto(self, r_cut, n_max, l_max):
        """Used to calculate the alpha and beta prefactors for the gto-radial
        basis.

        Args:
            r_cut(float): Radial cutoff.
            n_max(int): Number of gto radial bases.

        Returns:
            (np.ndarray, np.ndarray): The alpha and beta prefactors for all bases
            up to a fixed size of l=10.
        """
        # These are the values for where the different basis functions should decay
        # to: evenly space between 1 angstrom and r_cut.
        a = np.linspace(1, r_cut, n_max)
        threshold = 1e-3  # This is the fixed gaussian decay threshold

        alphas_full = np.zeros((l_max + 1, n_max))
        betas_full = np.zeros((l_max + 1, n_max, n_max))

        for l in range(0, l_max + 1):
            # The alphas are calculated so that the GTOs will decay to the set
            # threshold value at their respective cutoffs
            alphas = -np.log(threshold / np.power(a, l)) / a**2

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
                    "radial basis functions (n_max) or increasing the radial "
                    "cutoff (r_cut) is advised."
                )

            alphas_full[l, :] = alphas
            betas_full[l, :, :] = betas

        return alphas_full, betas_full

    def get_basis_poly(self, r_cut, n_max):
        """Used to calculate discrete vectors for the polynomial basis functions.

        Args:
            r_cut(float): Radial cutoff.
            n_max(int): Number of polynomial radial bases.

        Returns:
            (np.ndarray, np.ndarray): Tuple containing the evaluation points in
            radial direction as the first item, and the corresponding
            orthonormalized polynomial radial basis set as the second item.
        """
        # Calculate the overlap of the different polynomial functions in a
        # matrix S. These overlaps defined through the dot product over the
        # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
        # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
        # the basis orthonormal are given by B=S^{-1/2}
        S = np.zeros((n_max, n_max), dtype=np.float64)
        for i in range(1, n_max + 1):
            for j in range(1, n_max + 1):
                S[i - 1, j - 1] = (2 * (r_cut) ** (7 + i + j)) / (
                    (5 + i + j) * (6 + i + j) * (7 + i + j)
                )

        # Get the beta factors that orthonormalize the set with Löwdin
        # orthonormalization
        betas = sqrtm(np.linalg.inv(S))

        # If the result is complex, the calculation is currently halted.
        if betas.dtype == np.complex128:
            raise ValueError(
                "Could not calculate normalization factors for the radial "
                "basis in the domain of real numbers. Lowering the number of "
                "radial basis functions (n_max) or increasing the radial "
                "cutoff (r_cut) is advised."
            )

        # The radial basis is integrated in a very specific nonlinearly spaced
        # grid given by rx
        x = np.zeros(100)
        x[0] = -0.999713726773441234
        x[1] = -0.998491950639595818
        x[2] = -0.996295134733125149
        x[3] = -0.99312493703744346
        x[4] = -0.98898439524299175
        x[5] = -0.98387754070605702
        x[6] = -0.97780935848691829
        x[7] = -0.97078577576370633
        x[8] = -0.962813654255815527
        x[9] = -0.95390078292549174
        x[10] = -0.94405587013625598
        x[11] = -0.933288535043079546
        x[12] = -0.921609298145333953
        x[13] = -0.90902957098252969
        x[14] = -0.895561644970726987
        x[15] = -0.881218679385018416
        x[16] = -0.86601468849716462
        x[17] = -0.849964527879591284
        x[18] = -0.833083879888400824
        x[19] = -0.815389238339176254
        x[20] = -0.79689789239031448
        x[21] = -0.77762790964949548
        x[22] = -0.757598118519707176
        x[23] = -0.736828089802020706
        x[24] = -0.715338117573056447
        x[25] = -0.69314919935580197
        x[26] = -0.670283015603141016
        x[27] = -0.64676190851412928
        x[28] = -0.622608860203707772
        x[29] = -0.59784747024717872
        x[30] = -0.57250193262138119
        x[31] = -0.546597012065094168
        x[32] = -0.520158019881763057
        x[33] = -0.493210789208190934
        x[34] = -0.465781649773358042
        x[35] = -0.437897402172031513
        x[36] = -0.409585291678301543
        x[37] = -0.380872981624629957
        x[38] = -0.351788526372421721
        x[39] = -0.322360343900529152
        x[40] = -0.292617188038471965
        x[41] = -0.26258812037150348
        x[42] = -0.23230248184497397
        x[43] = -0.201789864095735997
        x[44] = -0.171080080538603275
        x[45] = -0.140203137236113973
        x[46] = -0.109189203580061115
        x[47] = -0.0780685828134366367
        x[48] = -0.046871682421591632
        x[49] = -0.015628984421543083
        x[50] = 0.0156289844215430829
        x[51] = 0.046871682421591632
        x[52] = 0.078068582813436637
        x[53] = 0.109189203580061115
        x[54] = 0.140203137236113973
        x[55] = 0.171080080538603275
        x[56] = 0.201789864095735997
        x[57] = 0.23230248184497397
        x[58] = 0.262588120371503479
        x[59] = 0.292617188038471965
        x[60] = 0.322360343900529152
        x[61] = 0.351788526372421721
        x[62] = 0.380872981624629957
        x[63] = 0.409585291678301543
        x[64] = 0.437897402172031513
        x[65] = 0.465781649773358042
        x[66] = 0.49321078920819093
        x[67] = 0.520158019881763057
        x[68] = 0.546597012065094168
        x[69] = 0.572501932621381191
        x[70] = 0.59784747024717872
        x[71] = 0.622608860203707772
        x[72] = 0.64676190851412928
        x[73] = 0.670283015603141016
        x[74] = 0.693149199355801966
        x[75] = 0.715338117573056447
        x[76] = 0.736828089802020706
        x[77] = 0.75759811851970718
        x[78] = 0.77762790964949548
        x[79] = 0.79689789239031448
        x[80] = 0.81538923833917625
        x[81] = 0.833083879888400824
        x[82] = 0.849964527879591284
        x[83] = 0.866014688497164623
        x[84] = 0.881218679385018416
        x[85] = 0.89556164497072699
        x[86] = 0.90902957098252969
        x[87] = 0.921609298145333953
        x[88] = 0.933288535043079546
        x[89] = 0.94405587013625598
        x[90] = 0.953900782925491743
        x[91] = 0.96281365425581553
        x[92] = 0.970785775763706332
        x[93] = 0.977809358486918289
        x[94] = 0.983877540706057016
        x[95] = 0.98898439524299175
        x[96] = 0.99312493703744346
        x[97] = 0.99629513473312515
        x[98] = 0.998491950639595818
        x[99] = 0.99971372677344123

        rx = r_cut * 0.5 * (x + 1)

        # Calculate the value of the orthonormalized polynomial basis at the rx
        # values
        fs = np.zeros([n_max, len(x)])
        for n in range(1, n_max + 1):
            fs[n - 1, :] = (r_cut - np.clip(rx, 0, r_cut)) ** (n + 2)

        gss = np.dot(betas, fs)

        return rx, gss
