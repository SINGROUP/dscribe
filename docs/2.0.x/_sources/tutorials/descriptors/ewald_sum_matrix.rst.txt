Ewald sum matrix
================

Ewald sum matrix :cite:`sm` can be viewed as a logical extension of the Coulomb
matrix for periodic systems, as it models the interaction between atoms in a
periodic crystal through electrostatic interaction. However, in order to
calculate the Coulomb interaction between the atomic cores in a periodic
crystal, the Ewald summation technique has to be used together with a uniform
neutralizing background charge.

Notice that the terms of the Ewald sum matrix in DScribe differ slightly from
the original article. We correct an issue in the energy term related to the
self-energy and the charged-cell correction. In the original article :cite:`sm`
(equation 20) this correction for the off-diagonal elements was defined as:

.. math::
    \phi^{\text{self}}_{ij} + \phi^{\text{bg}}_{ij} = -\frac{\alpha}{\sqrt{\pi}}(Z_i^2 + Z_j^2) -\frac{\pi}{2V\alpha^2}(Z_i + Z_j)^2~\forall~i \neq j

This term does however not correspond to the correct Ewald energy, as seen in
the case of two interacting atoms, one of which has no charge: :math:`Z_i = 0`,
:math:`Z_j \neq 0`. In this case the interaction should be zero, but the given
term is non-zero and additionally depends on the screening parameter
:math:`\alpha`.  DScribe instead uses the corrected term:

.. math::
    \phi^{\text{self}}_{ij} + \phi^{\text{bg}}_{ij} = -\frac{\pi}{2 V \alpha^2} Z_i Z_j~\forall~i \neq j

Setup
-----

Instantiating the object that is used to create Ewald sum matrices can be done as
follows:

.. literalinclude:: ../../../../examples/ewaldsummatrix.py
   :language: python
   :lines: 7-17

The constructor takes the following parameters:

.. automethod:: dscribe.descriptors.ewaldsummatrix.EwaldSumMatrix.__init__

Creation
--------

After the Ewald sum matrix has been set up, it may be used on periodic atomic
structures with the :meth:`~.EwaldSumMatrix.create`-method.

.. literalinclude:: ../../../../examples/ewaldsummatrix.py
   :start-after: Creation
   :language: python
   :lines: 1-14

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.ewaldsummatrix.EwaldSumMatrix.create
   :noindex:

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
examples are also available in dscribe/examples/ewaldsummatrix.py.

Accuracy
~~~~~~~~
Easiest way to control the accuracy of the Ewald summation is to use the
*accuracy*-parameter. Lower values of this parameter correspond to tighter
convergence criteria and better accuracy.

.. literalinclude:: ../../../../examples/ewaldsummatrix.py
   :start-after: Accuracy
   :language: python
   :lines: 4-5

Another option is to directly provide the real- and reciprocal space cutoffs:

.. literalinclude:: ../../../../examples/ewaldsummatrix.py
   :start-after: Accuracy
   :language: python
   :lines: 8

Total electrostatic energy
~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's calculate the electrostatic energy of a crystal by using the information
contained in the Ewald sum matrix.

.. literalinclude:: ../../../../examples/ewaldsummatrix.py
   :start-after: Energy
   :language: python
   :lines: 2-21
