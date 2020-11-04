Coulomb Matrix
==============

Coulomb Matrix (CM) :cite:`cm` is a simple global descriptor which mimics the
electrostatic interaction between nuclei.

Coulomb matrix is calculated with the equation below.

.. math::
    \begin{equation}
    M_{ij}^\mathrm{Coulomb}=\left\{
        \begin{matrix}
        0.5 Z_i^{2.4} & \text{for } i = j \\
            \frac{Z_i Z_j}{R_{ij}} & \text{for } i \neq j
        \end{matrix}
        \right.
    \end{equation}

The diagonal elements can be seen as the interaction of an atom with itself and
are essentially a polynomial fit of the atomic energies to the nuclear charge
:math:`Z_i`. The off-diagonal elements represent the Coulomb repulsion between
nuclei :math:`i` and :math:`j`.

Let's have a look at the CM for methanol:

.. image:: /_static/img/methanol-3d-balls.png
   :width: 344px
   :height: 229px
   :scale: 50 %
   :alt: image of methanol
   :align: center

.. math::
    \begin{bmatrix}
    36.9 & 33.7 & 5.5  & 3.1  & 5.5  & 5.5 \\
    33.7 & 73.5 & 4.0  & 8.2  & 3.8  & 3.8  \\
     5.5 & 4.0  & 0.5  & 0.35 & 0.56 & 0.56 \\
     3.1 & 8.2  & 0.35 & 0.5  & 0.43 & 0.43 \\
     5.5 & 3.8  & 0.56 & 0.43 & 0.5  & 0.56 \\
     5.5 & 3.8  & 0.56 & 0.43 & 0.56 & 0.5
    \end{bmatrix}

In the matrix above the first row corresponds to carbon (C) in methanol
interacting with all the other atoms (columns 2-5) and itself (column 1).
Likewise, the first column displays the same numbers, since the matrix is
symmetric. Furthermore, the second row (column) corresponds to oxygen and the
remaining rows (columns) correspond to hydrogen (H). Can you determine which
one is which?

Since the Coulomb Matrix was published in 2012 more sophisticated descriptors
have been developed. However, CM still does a reasonably good job when
comparing molecules with each other. Apart from that, it can be understood
intuitively and is a good introduction to descriptors.

Setup
-----

Instantiating the object that is used to create Coulomb matrices can be done as
follows:

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :language: python
   :lines: 1-11

The constructor takes the following parameters:

.. automethod:: dscribe.descriptors.coulombmatrix.CoulombMatrix.__init__

Creation
--------

After CM has been set up, it may be used on atomic structures with the
:meth:`~.CoulombMatrix.create`-method.

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :start-after: Creation
   :language: python
   :lines: 1-15

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.coulombmatrix.CoulombMatrix.create

Note that if you specify in *n_atoms_max* a lower number than atoms in your
structure it will cause an error. The output will in this case be a flattened
matrix, specifically a numpy array with size #atoms * #atoms. The number of
features may be requested beforehand with the
:meth:`~.MatrixDescriptor.get_number_of_features`-method.

In the case of multiple samples, the creation can also be parallellized by using the
*n_jobs*-parameter. This splits the list of structures into equally sized parts
and spaws a separate process to handle each part.

Examples
--------

The following examples demonstrate usage of the descriptor. These
examples are also available in dscribe/examples/coulombmatrix.py.

No flattening
~~~~~~~~~~~~~
You can control whether the returned array is two-dimensional or
one-dimensional by using the *flatten*-parameter

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :language: python
   :start-after: No flattening
   :lines: 1-7

No Sorting
~~~~~~~~~~~
By default, CM is sorted by the L2-norm (more on that later). In order to get
the unsorted CM it is necessary to specify the keyword *permutation = "none"*
when setting it up.

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :language: python
   :start-after: No sorting
   :lines: 1-8

Zero-padding
~~~~~~~~~~~~~
The number of features in CM depends on the size of the system. Since most
machine learning methods require size-consistent inputs it is convenient to
define the maximum number of atoms *n_atoms_max* in a dataset. If the structure
has fewer atoms, the rest of the CM will be zero-padded. One can imagine
non-interacting ghost atoms as place-holders to ensure the same number of atoms
in every system.

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :language: python
   :start-after: Zero-padding
   :lines: 1-7

Not meant for periodic systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The CM was not designed for periodic systems. If you do add periodic boundary
conditions, you will see that it does not change the elements.

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :language: python
   :start-after: Not meant for periodic systems
   :lines: 1-12

Instead, the :doc:`Sine Matrix <sine_matrix>` and the `Ewald Matrix
<ewald_matrix>` have been designed as periodic counterparts to the CM.

Invariance
-----------
A good descriptor should be invariant with respect to translation, rotation and
permutation. No matter how you translate or rotate it or change the indexing of
the atoms (not the atom types!), it will still be the same molecule! The
following lines confirm that this is true for CM.

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :language: python
   :start-after: Invariance
   :lines: 1-20

Options for permutation
-----------------------
The following snippet introduces the different options for handling permutation
invariance. See :cite:`cm_versions` for more information on these methods.

.. literalinclude:: ../../../../examples/coulombmatrix.py
   :language: python
   :start-after: No sorting
   :lines: 1-37

- **sorted_l2 (default)**: Sorts rows and columns by their L2-norm.
- **none**: keeps the order of the rows and columns as the atoms are read from
  the ase object.
- **random**: The term random can be misleading at first sight because it does not
  scramble the rows and columns completely randomly. The rows and columns are
  sorted by their L2-norm  after applying Gaussian noise to the norms. The
  standard deviation of the noise is determined by the additionally required
  *sigma*-parameter. *sigma* determines the standard deviation of the gaussian
  distributed noise determining how much the rows and columns of the randomly
  sorted matrix are scrambled. Feel free to try different *sigma* values to
  see the effect on the ordering. Optionally, you can specify a random *seed*.
  *sigma* and *seed* are ignored if *permutation* is other than "random". This
  option is useful if you want to augment your dataset, similar to augmented
  image datasets where each image gets mirrored, rotated, cropped or otherwise
  transformed. You would need to create several instances of the randomly
  sorted CM in a loop. The advantage of augmenting data like this over using
  completely random CM lies in the lower number of "likely permutations". Rows
  and columns of the CM are allowed to flip just so that the feature space
  (all possible CM) is smooth but also compact.
- **eigenspectrum**: Only the eigenvalues of the matrix are returned sorted by
  their absolute value in descending order. On one hand, it is a more compact
  descriptor, but on the other hand, it potentially loses information encoded
  in the CM interactions.

.. bibliography:: ../../references.bib
   :style: unsrt
   :filter: docname in docnames
