Smooth Overlap of Atomic Positions
==================================

Smooth Overlap of Atomic Positions (SOAP) is a descriptor that encodes regions
of atomic geometries by using a local expansion of a gaussian smeared atomic
density with orthonormal functions based on spherical harmonics and radial
basis functions.

The SOAP output from DScribe is the partial power spectrum vector
:math:`\mathbf{p}`, where the elements are defined as :cite:`soap2`

.. math::
   p^{Z_1 Z_2}_{n n' l} = \pi \sqrt{\frac{8}{2l+1}}\sum_m {c^{Z_1}_{n l m}}^*c^{Z_2}_{n' l m}

where :math:`n` and :math:`n'` are indices for the different radial basis
functions up to :math:`n_\mathrm{max}`, :math:`l` is the angular degree of the
spherical harmonics up to :math:`l_\mathrm{max}` and :math:`Z_1` and :math:`Z_2`
are atomic species.

The coefficients :math:`c^Z_{nlm}` are defined as the following inner
products:

.. math::
   c^Z_{nlm} =\iiint_{\mathcal{R}^3}\mathrm{d}V g_{n}(r)Y_{lm}(\theta, \phi)\rho^Z(\mathbf{r}).

where :math:`\rho^Z(\mathbf{r})` is the gaussian smoothed atomic density for
atoms with atomic number :math:`Z` defined as

.. math::
   \rho^Z(\mathbf{r}) = \sum_i^{\lvert Z_i \rvert} e^{-1/2\sigma^2 \lvert \mathbf{r} - \mathbf{R}_i \rvert^2}

:math:`Y_{lm}(\theta, \phi)` are the real spherical harmonics, and
:math:`g_{n}(r)` is the radial basis function.

For the radial degree of freedom the selection of the basis function
:math:`g_{n}(r)` is not as trivial and multiple approaches may be used. By
default the DScribe implementation uses spherical gaussian type orbitals as
radial basis functions :cite:`akisoap`, as they allow much faster analytic
computation. We however also include the possibility of using the original
polynomial radial basis set :cite:`soap1`.

The spherical harmonics definition used by DScribe is based on `real (tesseral)
spherical harmonics
<https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form>`_. This real form
spans the same space as the complex version, and is defined as a linear
combination of the complex basis. As the atomic density is a real-valued
quantity (no imaginary part) it is natural and computationally easier to use
this form that does not require complex algebra.

The SOAP kernel :cite:`soap1` between two atomic environments can be retrieved
as a normalized polynomial kernel of the partial powers spectrums:

.. math::
   K^\mathrm{SOAP}(\mathbf{p}, \mathbf{p'}) = \left( \frac{\mathbf{p} \cdot \mathbf{p'}}{\sqrt{\mathbf{p} \cdot \mathbf{p}~\mathbf{p'} \cdot \mathbf{p'}}}\right)^{\xi}

Although this is the original similarity definition, nothing in practice
prevents the usage of the output in non-kernel based methods or with other
kernel definitions.

.. note::
   Notice that by default the SOAP output by DScribe only contains unique
   terms, i.e. some terms that would be repeated multiple times due to symmetry
   are left out. In order to *exactly* match the original SOAP kernel
   definition, you have to double the weight of the terms which have a
   symmetric counter part in :math:`\mathbf{p}`.

The partial SOAP spectrum ensures stratification of the output by species and
also provides information about cross-species interaction. See the
:meth:`~.SOAP.get_location` method for a way of easily accessing parts of the
output that correspond to a particular species combination. In pseudo-code the
ordering of the output vector is as follows:

.. code-block:: none

   for Z in atomic numbers in increasing order:
      for Z' in atomic numbers in increasing order:
         for l in range(l_max+1):
            for n in range(n_max):
               for n' in range(n_max):
                  if (n', Z') >= (n, Z):
                     append p(\chi)^{Z Z'}_{n n' l}` to output


Setup
-----

Instantiating the object that is used to create SOAP can be done as follows:

.. literalinclude:: ../../../../examples/soap.py
   :language: python
   :lines: 1-15

The constructor takes the following parameters:

.. automethod:: dscribe.descriptors.soap.SOAP.__init__

Increasing the arguments *nmax* and *lmax* makes SOAP more accurate but also
increases the number of features.

Creation
--------
After SOAP has been set up, it may be used on atomic structures with the
:meth:`~.SOAP.create`-method.

.. literalinclude:: ../../../../examples/soap.py
   :start-after: Creation
   :language: python
   :lines: 1-16

As SOAP is a local descriptor, it also takes as input a list of atomic indices
or positions. If no such positions are defined, SOAP will be created for each
atom in the system. The call syntax for the create-method is as follows:

.. automethod:: dscribe.descriptors.soap.SOAP.create
   :noindex:

The output will in this case be a numpy array with shape [#positions,
#features]. The number of features may be requested beforehand with the
:meth:`~.SOAP.get_number_of_features`-method.

Examples
--------
The following examples demonstrate common use cases for the descriptor. These
examples are also available in dscribe/examples/soap.py.

Finite systems
~~~~~~~~~~~~~~
Adding SOAP to water is as easy as:

.. literalinclude:: ../../../../examples/soap.py
   :language: python
   :lines: 18-27

We are expecting a matrix where each row represents the local environment of
one atom of the molecule. The length of the feature vector depends on the
number of species defined in *species* as well as *nmax* and *lmax*. You can
try by changing *nmax* and *lmax*.

.. literalinclude:: ../../../../examples/soap.py
   :language: python
   :lines: 35-40

Periodic systems
~~~~~~~~~~~~~~~~

Crystals can also be SOAPed by simply setting :code:`periodic=True` in the SOAP
constructor and ensuring that the :code:`ase.Atoms` objects have a unit cell
and their periodic boundary conditions are set with the :code:`pbc`-option. 

.. literalinclude:: ../../../../examples/soap.py
   :language: python
   :start-after: Periodic systems
   :lines: 1-17

Since the SOAP feature vectors of each of the four copper atoms in the cubic
unit cell match, they turn out to be equivalent.

.. _soap-weighting:

Weighting
~~~~~~~~~

The default SOAP formalism weights the atomic density equally no matter how far
it is from the the position of interest. Especially in systems with uniform
atomic density this can lead to the atoms in farther away regions
dominating the SOAP spectrum. It has been shown :cite:`soap_weighting` that radially
scaling the atomic density can help in creating a suitable balance that gives
more importance to the closer-by atoms. This idea is very similar to the
weighting done in the MBTR descriptor.

The weighting could be done by directly adding a weighting function
:math:`w(r)` in the integrals:

.. math::
   c^Z_{nlm} =\iiint_{\mathcal{R}^3}\mathrm{d}V g_{n}(r)Y_{lm}(\theta, \phi)w(r)\rho^Z(\mathbf{r}).

This can, however, complicate the calculation of these integrals considerably.
Instead of directly weighting the atomic density, we can weight the
contribution of individual atoms by scaling the amplitude of their Gaussian
contributions:

.. math::
   \rho^Z(\mathbf{r}) = \sum_i^{\lvert Z_i \rvert} w(\lvert \mathbf{R}_i \rvert)e^{-1/2\sigma^2 \lvert \mathbf{r} - \mathbf{R}_i \rvert^2}

This approximates the "correct" weighting very well as long as the width of the
atomic Gaussians (as determined by ``sigma``) is small compared to the
variation in the weighting function :math:`w(r)`.

DScribe currently supports this latter simplified weighting, with different
weighting functions, and a possibility to also separately weight the central
atom (sometimes the central atom will not contribute meaningful information and
you may wish to even leave it out completely by setting :code:`w0=0`). Three
different weighting functions are currently supported, and some example
instances from these functions are plotted below.

.. figure:: /_static/img/soap_weighting.png
   :width: 700px
   :alt: SOAP weighting
   :align: center

   Example instances of weighting functions defined on the interval [0, 1]. The
   ``poly`` function decays exactly to zero at :math:`r=r_0`, the others
   decay smoothly towards zero.

When using a weighting function, you typically will also want to restrict
``r_cut`` into a range that lies within the domain in which your weighting
function is active. You can achieve this by manually tuning r_cut to a range
that fits your weighting function, or if you set :code:`r_cut=None`, it will be
set automatically into a sensible range which depends on your weighting
function. You can see more details and the algebraic form of the weighting
functions in the constructor documentation.

Locating information
~~~~~~~~~~~~~~~~~~~~
The SOAP class provides the :meth:`~.SOAP.get_location`-method. This method can
be used to query for the slice that contains a specific element combination.
The following example demonstrates its usage.

.. literalinclude:: ../../../../examples/soap.py
   :start-after: Locations
   :language: python
   :lines: 1-8

Sparse output
~~~~~~~~~~~~~

For more information on the reasoning behind sparse output and its usage check
our separate :doc:`documentation on sparse output <../sparse>`.
Enabling sparse output on SOAP is as easy as setting :code:`sparse=True`:

.. literalinclude:: ../../../../examples/soap.py
   :language: python
   :start-after: Sparse output
   :lines: 1-19

Average output
~~~~~~~~~~~~~~

One way of turning a local descriptor into a global descriptor is simply by
taking the average over all sites. DScribe supports two averaging modes:
*inner* and *outer*. The inner average is taken over the sites before summing
up the magnetic quantum number. Outer averaging instead averages over the
power spectrum of individual sites. In general, the inner averaging will
preserve the configurational information better but you can experiment with
both versions.

.. literalinclude:: ../../../../examples/soap.py
   :language: python
   :start-after: Average output
   :lines: 1-19

The result will be a feature vector and not a matrix, so it no longer depends
on the system size. This is necessary to compare two or more structures with
different number of elements. We can do so by e.g. applying the distance metric
of our choice.

.. literalinclude:: ../../../../examples/soap.py
   :language: python
   :start-after: Distance
   :lines: 1-7

It seems that the local environments of water and hydrogen peroxide are more
similar to each other. To see more advanced methods for comparing structures of
different sizes with each other, see the :doc:`kernel building tutorial
</tutorials/similarity_analysis/kernels>`. Notice that simply averaging the SOAP vector does not always
correspond to the Average Kernel discussed in the kernel building tutorial, as
for non-linear kernels the order of kernel calculation and averaging matters.

.. bibliography:: ../../references.bib
   :style: unsrt
   :filter: docname in docnames
