Many-body Tensor Representation
===============================
The many-body tensor representation (MBTR) :cite:`mbtr` encodes a structure by
using a distribution of different structural motifs. It can be used directly
for both finite and periodic systems.

In MBTR a geometry function :math:`g_k` is used to transform a chain of :math:`k` atoms,
into a single scalar value. The distribution of these scalar values is then
constructed with kernel density estimation to represent the structure.

.. figure:: /_static/img/mbtr.jpg
   :width: 1144px
   :height: 772px
   :scale: 40 %
   :alt: MBTR stratification
   :align: center

   Illustration of the MBTR output for a water molecule.

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
:meth:`~.MBTR.create`-method.

.. literalinclude:: ../../../examples/mbtr.py
   :language: python
   :lines: 17-26

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.mbtr.MBTR.create

The output will in this case be a numpy array with shape [#positions,
#features]. The number of features may be requested beforehand with the
:meth:`~.MBTR.get_number_of_features`-method.

Examples
--------
The following examples demonstrate common use cases for the descriptor. These
examples are also available in dscribe/examples/mbtr.py.

Finite systems
~~~~~~~~~~~~~~
For finite systems we have to specify *periodic=False* in the constructor. The
need to apply weighting depends on the size of the system: for small systems,
such as small molecules, the benefits are small. However for larger systems,
such as clusters and bigger molecules, adding weighting will help in removing
"noise" coming from atom combinations that are physically very far apart and do
not have any meaningful direct interaction in the system. The following code
demonstrates both approaches.

Periodic systems
~~~~~~~~~~~~~~~~
When applying MBTR to periodic systems, use *periodic=True* in the constructor.
For periodic systems a weighting function needs to be defined, and an exception
is raised if it not given. The weighting essentially determines how many
periodic copies of the cell we need to include in the calculation, and without
it we would not know when to stop the periodic repetition. The following code
demonstrate how to apply MBTR on a periodic crystal.

A problem with periodic crystals that is not directly solved by the MBTR
formalism is the fact that multiple different cells shapes and sizes can be
used for the same crystal. Different shapes and sizes will have the same
looking MBTR output, but will be scaled differently.

Visualization
~~~~~~~~~~~~~
The MBTR output vector can be visualized easily. The following snippet
demonstrates how the output for :math:`k=2` can be visualized with matplotlib.

.. literalinclude:: ../../../examples/mbtr.py
   :language: python
   :lines: 36-

.. bibliography:: ../references.bib
   :style: unsrt
   :filter: docname in docnames
