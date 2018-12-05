Many-body Tensor Representation
===============================
The many-body tensor representation (MBTR) encodes a structure by using a
distribution of different structural motifs. It can be used directly for both
finite and periodic systems.

In MBTR a geometry function :math:`g_k` is used to transform a chain of :math:`k` atoms,
into a single scalar value. The distribution of these scalar values is then
constructed with kernel density estimation to represent the structure.

.. figure:: /_static/img/mbtr.jpg
   :scale: 50 %
   :alt: MBTR stratification
   :align: center

   Illustration of the MBTR output for a water molecule.

For more details see the original paper: `Unified Representation for Machine
Learning of Molecules and Crystals, Haoyan Huo and Matthias Rupp,
arXiv preprint, (2017)
<http://arxiv.org/abs/1704.06439>`_

Setup
-----

Instantiating an MBTR descriptor can be done as follows:

.. literalinclude:: ../../../examples/mbtr.py
   :language: python
   :lines: 1-15

The arguments have the following effect:

.. automethod:: dscribe.descriptors.mbtr.MBTR.__init__


Creation
--------
After MBTR has been set up, it may be used on atomic structures with the
:meth:`.MBTR.create`-function.

.. literalinclude:: ../../../examples/mbtr.py
   :language: python
   :lines: 17-26

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.mbtr.MBTR.create

The output will in this case be a numpy array with shape [#positions,
#features]. The number of features may be requested beforehand with the
:meth:`.MBTR.get_number_of_features`-function.

Examples
--------

Finite systems
~~~~~~~~~~~~~~


Periodic systems
~~~~~~~~~~~~~~~~


Sparse output
~~~~~~~~~~~~~


Working on multiple samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~


Hyperparameter optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~


Visualization
~~~~~~~~~~~~~
The MBTR output vector can also be visualized easily. The following snippet
demonstrates how the output for :math:`k=2` can be visualized with
matplotlib.

.. literalinclude:: ../../../examples/mbtr.py
   :language: python
   :lines: 36-
