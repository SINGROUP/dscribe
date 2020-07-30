Smooth Overlap of Atomic Positions
==================================

Smooth Overlap of Atomic Positions (SOAP) is a descriptor that encodes regions
of atomic geometries by using a local expansion of a gaussian smeared atomic
density with orthonormal functions based on spherical harmonics and radial
basis functions.

The SOAP output from DScribe is the partial power spectrum vector
:math:`\mathbf{p}(\mathbf{r})`, where the elements are defined as :cite:`soap2`

.. math::
   p(\mathbf{r})^{Z_1 Z_2}_{n n' l} = \pi \sqrt{\frac{8}{2l+1}}\sum_m c^{Z_1}_{n l m}(\mathbf{r})^*c^{Z_2}_{n' l m}(\mathbf{r})

where :math:`n` and :math:`n'` are indices for the different radial basis
functions up to :math:`n_\mathrm{max}`, :math:`l` is the angular degree of the
spherical harmonics up to :math:`l_\mathrm{max}` and :math:`Z_1` and :math:`Z_2`
are atomic species.

The coefficients :math:`c^Z_{nlm}` are defined as the following inner
products:

.. math::
   c^Z_{nlm}(\mathbf{r}) =\iiint_{\mathcal{R}^3}\mathrm{d}V g_{n}(r)Y_{lm}(\theta, \phi)\rho^Z(\mathbf{r}).

where :math:`\mathbf{r}` is a position in space, :math:`\rho^Z(\mathbf{r})` is
the gaussian smoothed atomic density for atoms with atomic number :math:`Z`,
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

.. literalinclude:: ../../../examples/soap.py
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

.. literalinclude:: ../../../examples/soap.py
   :start-after: Creation
   :language: python
   :lines: 1-16

As SOAP is a local descriptor, it also takes as input a list of atomic indices
or positions. If no such positions are defined, SOAP will be created for each
atom in the system. The call syntax for the create-method is as follows:

.. automethod:: dscribe.descriptors.soap.SOAP.create

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

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 18-27

We are expecting a matrix where each row represents the local environment of
one atom of the molecule. The length of the feature vector depends on the
number of species defined in *species* as well as *nmax* and *lmax*. You can
try by changing *nmax* and *lmax*.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 35-40

Periodic systems
~~~~~~~~~~~~~~~~

Crystals can also be SOAPed by simply setting the *periodic* keyword to True.
In this case a cell needs to be defined for the ase object.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :start-after: Periodic systems
   :lines: 1-17

Since the SOAP feature vectors of each of the four copper atoms in the cubic
unit cell match, they turn out to be equivalent.

Locating information
~~~~~~~~~~~~~~~~~~~~
The SOAP class provides the :meth:`~.SOAP.get_location`-method. This method can
be used to query for the slice that contains a specific element combination.
The following example demonstrates its usage.

.. literalinclude:: ../../../examples/soap.py
   :start-after: Locations
   :language: python
   :lines: 1-8

Sparse output
~~~~~~~~~~~~~

If the descriptor size is large (this can be the case for instance with a
myriad of different element types as well as high *nmax* and *lmax*) more often
than not considerable parts of the features will be zero. In this case saving
the results in a sparse matrix will save memory. DScribe does so by default
using the `scipy-library
<https://docs.scipy.org/doc/scipy/reference/sparse.html>`_. Be aware between
the different types:

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :start-after: Sparse output
   :lines: 1-19

Most operations work on sparse matrices as they would on numpy matrices.
Otherwise, a sparse matrix can simply be converted calling the *.toarray()*
method. For further information check the `scipy documentation
<https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ on sparse matrices.

Average output
~~~~~~~~~~~~~~

One way of turning a local descriptor into a global descriptor is simply by
taking the average over all sites. DScribe supports two averaging modes:
*inner* and *outer*. The inner average is taken over the sites before summing
up the magnetic quantum number. Outer averaging instead averages over the
power spectrum of individual sites. In general, the inner averaging will
preserve the configurational information better but you can experiment with
both versions.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :start-after: Average output
   :lines: 1-19

The result will be a feature vector and not a matrix, so it no longer depends
on the system size. This is necessary to compare two or more structures with
different number of elements. We can do so by e.g. applying the distance metric
of our choice.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :start-after: Distance
   :lines: 1-7

It seems that the local environments of water and hydrogen peroxide are more
similar to each other. To see more advanced methods for comparing structures of
different sizes with each other, see the :doc:`kernel building tutorial
<kernels>`. Notice that simply averaging the SOAP vector does not always
correspond to the Average Kernel discussed in the kernel building tutorial, as
for non-linear kernels the order of kernel calculation and averaging matters.

.. bibliography:: ../references.bib
   :style: unsrt
   :filter: docname in docnames
