Coulomb Matrix
==============

Coulomb Matrix (CM) is a simple global descriptor which mimics the electrostatic interaction between nuclei. The original article outlines the concept of the descriptor and the relation to its name giver coulomb repulsion.

`Rupp, M.; Tkatchenko, A.; Muller, K.-R.; von Lilienfeld, O. A.; Müller, K.-R.; Lilienfeld, V.; Anatole, O. Phys. Rev. Lett. 2012, 108 (5), 58301
<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301>`_



Setup
-----

In order to create a coulomb matrix, we first set up the descriptor and then create an instance of it on a specific atomic structure.

Instantiating a CM descriptor can be done as follows:

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 1-12

The arguments have the following effect:

.. automethod:: dscribe.descriptors.coulombmatrix.CoulombMatrix.__init__

Creation
--------

After CM has been set up, it may be used on atomic structures with the
:meth:`.CoulombMatrix.create`-function.

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 13-23

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.coulombmatrix.CoulombMatrix.create


Note that if you specify in *n_atoms_max* a lower number than atoms in your structure it will cause an error. The output will in this case be a flattened matrix, specifically a numpy array with size #atoms * #atoms. The number of features may be requested beforehand with the
:meth:`.CoulombMatrix.get_number_of_features`-function.


Background on Coulomb Matrix
----------------------------

Since the Coulomb Matrix was published in 2012 more sophisticated descriptors have been developed. However, CM still does a reasonably good job when comparing molecules with each other. Apart from that, it can be understood intuitively and is a good introduction to descriptors.

The CM is calculated with the equation below.

.. math::
    \begin{equation}
    M_{ij}=\left\{
                    \begin{matrix}
                    0.5 \cdot Z_i^{2.4} & \text{for }   i = j   \\
                     \frac{Z_i \cdot Z_j}{R_{ij}} & \text{for }   i \neq j
                    \end{matrix}
                  \right.
    \end{equation}

The diagonal elements can be seen as the interaction of an atom with itself and are essentially a polynomial fit of the atomic energies to the nuclear charge :math:`Z_i`. The off-diagonal elements represent the Coulomb repulsion between nuclei :math:`i` and :math:`j`.

Let's have a look at the CM of methanol:

image on methanol next to coulomb matrix

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

In the matrix above the first row corresponds to carbon (C) in methanol interacting with all the other atoms (columns 2-5) and itself (column 1). Likewise, the first column displays the same numbers, since the matrix is symmetric. Furthermore, the second row (column) corresponds to oxygen and the remaining rows (columns) correspond to hydrogen (H). Can you determine which one is which?

The ordering of the rows and columns is arbitrary. If oxygen appeared before carbon, the first and the second row and coulumn would have to be flipped. Let us reproduce the above matrix by switching off the *flatten* keyword.

No flattening
~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 24-33

No Sorting
~~~~~~~~~~~

You did not get the matrix in the same order. By default, CM is sorted by the L2-norm (more on that later). In order to get the unsorted CM it is necessary to specify the keyword *permutation* = "None" when setting it up.


.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 82-86


Zero-padding
~~~~~~~~~~~~~

The number of features in CM depends on the size of the system. Since most machine learning methods require size-consistent inputs it is convenient to define the maximum number of atoms *n_atoms_max* in a dataset. If the structure has fewer atoms, the rest of the CM will be zero-padded. One can imagine non-interacting ghost atoms as place-holders to ensure the same number of atoms in every system.

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 34-42


Not meant for periodic systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CM was not designed for periodic systems. If you do add periodic boundary conditions, you will see that it does not change the elements.

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 43-56

Instead, the :doc:`Sine Matrix <sine_matrix>` and the `Ewald Matrix <ewald_matrix>` have been designed as periodic counterparts to the CM.


Invariance
-----------

A good descriptor should be invariant with respect to translation, rotation and permutation. No matter how you translate or rotate it or change the indexing of the atoms (not the atom types!), it will still be the same molecule! The following lines confirm that this is true for CM.

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 60-79

Permutational invariance is obtained because it is sorted by default. The rows and columns are sorted in decending order with respect to the L2-norm.


Options for permutation
-----------------------

"None" keeps the order of the rows and columns as the atoms are read from the ase object. The default is "sorted_l2" which sorts rows and columns by their L2-norm. There are two more options: "random" and "eigenspectrum". The term random can be misleading at first sight because it does not scramble the rows and columns completely randomly.

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 82-105

The rows and columns are sorted by their L2-norm  after applying Gaussian noise to the norms. The standard deviation of the noise is determined by the additionally required *sigma*-parameter. *sigma* determines the standard deviation of the gaussian distributed noise determining how much the rows and columns of the randomly sorted matrix are scrambled. Feel free to try different *sigma* values to see the effect on the ordering.Optionally, you can specify a random *seed*. *sigma* and *seed* are ignored if *permutation* is other than "random".

This option is useful if you want to augment your dataset, similar to augmented image datasets where each image gets mirrored, rotated, cropped or otherwise transformed.
You would need to create several instances of the randomly sorted CM in a loop. The advantage of augmenting data like this over using completely random CM lies in the lower number of "likely permutations". Rows and columns of the CM are allowed to flip just so that the feature space (all possible CM) is smooth but also compact. For further reading, consult:

`Hansen, K.; Biegler, F.; Ramakrishnan, R.; Pronobis, W.; von Lilienfeld, O. A.; Müller, K.-R.; Tkatchenko, A. J. Phys. Chem. Lett. 2015, 6 (12), 2326–2331.
<https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.5b00831>`_


Eigenvector - a more compact descriptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last *permutation* option is strictly not a permutation, but rather a smaller descriptor. Here, only the eigenvalues of the matrix are returned sorted by their absolute value in descending order.

.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 109-117

On one hand, it is a more compact descriptor. On the other hand, it potentially destroys information encoded in the CM interactions.

Creating multiple descriptors in parallel
------------------------------------------

In machine learning we usually want to compare many structures with each other.
With the above functions, it is possible to iterate over a list of ase-ojects.
For convenience we provide the :func:`.create_batch` function that can be used
to build the output for multiple systems in parallel.


.. literalinclude:: ../../../examples/coulombmatrix.py
   :language: python
   :lines: 118-129

