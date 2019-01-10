Atom-centered Symmetry Functions
================================

Atom-centered Symmetry Functions (ACSFs) can be used to represent the local
environment near an atom by using a fingerprint composed of the output of
multiple two- and three-body functions that can be customized to detect
specific structural features.

For more details see the original paper: `Atom-centered symmetry functions for
constructing high-dimensional neural network potentials, Jörg Behler, J. Chem.
Phys. 134, 074106 (2011) <https://doi.org/10.1063/1.3553717>`_

Setup
-----

Instantiating an ACSF descriptor can be done as follows:

.. literalinclude:: ../../../examples/acsf.py
   :language: python
   :lines: 1-9

The arguments have the following effect:

.. automethod:: dscribe.descriptors.acsf.ACSF.__init__


Creation
--------
After ACSF has been set up, it may be used on atomic structures with the
:meth:`.ACSF.create`-function.

.. literalinclude:: ../../../examples/acsf.py
   :language: python
   :lines: 11-20

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.acsf.ACSF.create

The output will in this case be a numpy array with shape [#positions,
#features]. The number of features may be requested beforehand with the
:meth:`.ACSF.get_number_of_features`-function.
