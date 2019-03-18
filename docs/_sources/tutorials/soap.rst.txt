Smooth Overlap of Atomic Positions
==================================

Smooth Overlap of Atomic Positions (SOAP) is a descriptor that encodes regions
of atomic geometries by using a local expansion of a gaussian smeared atomic
density with orthonormal functions based on spherical harmonics.

The output is a vector :math:`\mathbf{p}(\mathbf{r})`, where the different elements are formed
from the partial SOAP power spectrum defined as :cite:`soap2`

.. math::
   p(\mathbf{r})^{Z Z'}_{n n' l} = \pi \sqrt{\frac{8}{2l+1}}\sum_m c^{Z}_{n l m}(\mathbf{r})^{\dagger}c^{Z'}_{n' l m}(\mathbf{r})

where :math:`\mathbf{r}` is a position in space, :math:`c^{Z}_{n l m}` are the
expansion coefficients of the gaussian smoothed atomic density at position
:math:`\mathbf{r}` (expanded in the basis of spherical harmonics and orthonormal
radial basis functions), :math:`n` and :math:`n'` are indices for different
radial basis functions up to :math:`n_\mathrm{max}`, :math:`l` is the angular
degree of spherical harmonics up to :math:`l_\mathrm{max}` and :math:`Z` and
:math:`Z'` are atomic species.  This form ensures stratification of the output
by species and also provides information about cross-species interaction.

In pseudo-code the ordering of the output vector is as follows:

.. code-block:: none

   for Z in atomic numbers in increasing order:
      for Z' in atomic numbers in increasing order:
         for l in range(l_max+1):
            for n in range(n_max):
               for n' in range(n_max):
                  if n' >= n and Z' >= Z:
                     append p(\chi)^{Z Z'}_{n n' l}` to output vector

By default the implementation uses spherical gaussian type orbitals as radial
basis functions :cite:`akisoap`, as they allow much faster analytic
computation. We however also include the possibility of using the original
polynomial radial basis set :cite:`soap1`.

Setup
-----

Instantiating a SOAP descriptor can be done as follows:

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 1-15

The arguments have the following effect:

.. automethod:: dscribe.descriptors.soap.SOAP.__init__

Since a smooth cutoff-function is used, sudden changes of configurations at the
cutoff distance are mitigated. Increasing the arguments *nmax* and *lmax*
makes SOAP more accurate but also increases the number of features.

Creation
--------
After SOAP has been set up, it may be used on atomic structures with the
:meth:`.SOAP.create`-function.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 17-23

As SOAP is a local descriptor, it also takes as input a list of atomic indices
or positions. If no such positions are defined, SOAP will be created for each
atom in the system. The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.soap.SOAP.create

The output will in this case be a numpy array with shape [#positions,
#features]. The number of features may be requested beforehand with the
:meth:`.SOAP.get_number_of_features`-function.

Examples
--------
The following examples demonstrate common use cases for the descriptor. These
examples are also available in dscribe/examples/soap.py.

Finite systems
~~~~~~~~~~~~~~
Let's first create an ase object for a water molecule:

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 18-20

Adding SOAP to the water is as easy as:

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 22-26

We are expecting a matrix where each row represents the local environment of
one atom of the molecule. The nth feature vector corresponds to the local SOAP
around the nth atom of the ase object. The length of the feature vector depends
on the number of species defined in *atomic_numbers* as well as *nmax* and
*lmax*. You can try by changing *nmax* and *lmax*.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 29-32

Let's SOAP another molecule. In order to compare water with methanol, one needs
to adapt the species in atomic_numbers first defined when initializing SOAP.
Adding carbon (atomic number = 6) will not affect the previous features of
water, it will merely add zero-padded regions to the feature vector.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 35-48

Periodic systems
~~~~~~~~~~~~~~~~

Crystals can also be SOAPed by simply setting the *periodic* keyword to True.
In this case a cell needs to be defined for the ase object.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 51-62

Since the SOAP feature vectors of each of the four copper atoms in the cubic
unit cell match, they turn out to be equivalent.

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
   :lines: 69-88

Most operations work on sparse matrices as they would on numpy matrices.
Otherwise, a sparse matrix can simply be converted calling the *.toarray()*
method. For further information check the `scipy documentation
<https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ on sparse matrices.

Average output
~~~~~~~~~~~~~~

One way of turning a local descriptor into a global descriptor is by taking the
average over all atoms. Since SOAP separates features by atom types, this
essentially means averaging over atoms of the same type.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 90-98

The result will be a feature vector and not a matrix, so it no longer depends
on the system size. This is necessary to compare two or more structures with
different number of elements. We can do so by e.g. applying the distance metric of
our choice.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 100-115

It seems that the local environments of water and hydrogen peroxide are more
similar to each other. To see more advanced methods for comparing structures of different
sizes with each other, see the :doc:`kernel building tutorial <kernels>`.

Working on multiple samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In machine learning we usually want to compare many structures with each other.
With the above functions, it is possible to iterate over a list of ase-ojects.
For convenience we provide the :func:`.create_batch` function that can be used
to build the output for multiple systems in parallel.

.. literalinclude:: ../../../examples/soap.py
   :language: python
   :lines: 117-121

Implementation note: If you specify *average = False*, you can only run one
local environment per molecule. That means you would need to specify one
location using the optional parameter positions. We are going to alleviate this
constraint soon. Try if it works now, the tutorial might not catch up with the
development!

.. bibliography:: ../references.bib
   :style: plain
