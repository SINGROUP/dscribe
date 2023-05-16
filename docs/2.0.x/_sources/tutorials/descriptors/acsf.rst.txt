Atom-centered Symmetry Functions
================================

Atom-centered Symmetry Functions (ACSFs) :cite:`acsf` can be used to represent
the local environment near an atom by using a fingerprint composed of the
output of multiple two- and three-body functions that can be customized to
detect specific structural features.

Notice that the DScribe output for ACSF does not include the species of the
central atom in any way. If the chemical identify of the central species is
important, you may want to consider a custom output stratification scheme based
on the chemical identity of the central species, or the inclusion of the
species identify in an additional feature. Training a separate model for each
central species is also possible.

The ACSF output is stratified by the species of the neighbouring atoms. In
pseudo-code the ordering of the output vector is as follows:

.. code-block:: none

   for Z in atomic numbers in increasing order:
       append value for G1 to output
       for G2 in g2_params:
           append value for G2 to output
       for G3 in g3_params:
           append value for G3 to output
   for Z in atomic numbers in increasing order:
      for Z' in atomic numbers in increasing order:
          if and Z' >= Z:
              for G4 in g4_params:
                  append value for G4 to output
              for G5 in g5_params:
                  append value for G5 to output

Setup
-----

Instantiating an ACSF descriptor can be done as follows:

.. literalinclude:: ../../../../examples/acsf.py
   :language: python
   :lines: 1-9

The arguments have the following effect:

.. automethod:: dscribe.descriptors.acsf.ACSF.__init__


Creation
--------
After ACSF has been set up, it may be used on atomic structures with the
:meth:`~.ACSF.create`-method.

.. literalinclude:: ../../../../examples/acsf.py
   :language: python
   :lines: 12-20

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.acsf.ACSF.create
   :noindex:

The output will in this case be a numpy array with shape :code:`[n_positions, n_features]`.
The number of features may be requested beforehand with the :meth:`~.ACSF.get_number_of_features`-method.

.. bibliography:: ../../references.bib
   :style: unsrt
   :filter: docname in docnames
