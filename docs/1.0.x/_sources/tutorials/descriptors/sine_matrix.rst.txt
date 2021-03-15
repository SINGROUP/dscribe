Sine matrix
===========

The sine matrix :cite:`sm` captures features of interacting atoms in a periodic
system with a very low computational cost. The matrix elements are defined by

.. math::
    \begin{equation}
    M_{ij}^\mathrm{sine}=\left\{
        \begin{matrix}
        0.5 Z_i^{2.4} & \text{for } i = j \\
            \frac{Z_i Z_j}{\lvert \mathbf{B} \cdot \sum_{k=\{x,y,z\}} \hat{\mathbf{e}}_k \sin^2 \left( \pi \mathbf{B}^{-1} \cdot \left( \mathbf{R}_{i} - \mathbf{R}_{j} \right) \right)\rvert} & \text{for } i \neq j
        \end{matrix}
        \right.
    \end{equation}

Here :math:`\mathbf{B}` is a matrix formed by the lattice vectors and
:math:`\hat{\mathbf{e}}_k` are the cartesian unit vectors. This functional form
has no physical interpretation, but it captures some of the properties of the
Coulomb interaction, such as the periodicity of the crystal lattice and an
infinite energy when two atoms overlap.

Setup
-----

Instantiating the object that is used to create Sine matrices can be done as
follows:

.. literalinclude:: ../../../../examples/sinematrix.py
   :language: python
   :lines: 1-9

The constructor takes the following parameters:

.. automethod:: dscribe.descriptors.sinematrix.SineMatrix.__init__

Creation
--------

After the Sine matrix has been set up, it may be used on periodic atomic
structures with the :meth:`~.SineMatrix.create`-method.

.. literalinclude:: ../../../../examples/sinematrix.py
   :start-after: Creation
   :language: python
   :lines: 1-14

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.sinematrix.SineMatrix.create

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
examples are also available in dscribe/examples/sinematrix.py.

Interaction in a periodic crystal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following code calculates the interaction value, as defined by the sine
matrix, between two aluminum atoms in an FCC-lattice. The values are calculated
in the xy-plane.

.. literalinclude:: ../../../../examples/sinematrix.py
   :start-after: Visualization
   :language: python
   :lines: 1-41

This code will result in the following plot:

.. figure:: /_static/img/sinematrix.png
   :width: 576
   :height: 480px
   :scale: 80 %
   :alt: sine matrix
   :align: center

   Illustration of the periodic interaction defined by the sine matrix.

From the figure one can see that the sine matrix correctly encodes the
periodicity of the lattice. Notice that the shape of the interaction is however
not perfectly spherical.

.. bibliography:: ../../references.bib
   :style: unsrt
   :filter: docname in docnames
