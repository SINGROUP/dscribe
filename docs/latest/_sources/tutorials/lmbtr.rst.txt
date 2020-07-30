Local Many-body Tensor Representation
=====================================
As the name suggests, the Local Many-body Tensor Representation (LMBTR) is a
modification of MBTR for local environments. It is advisable to first check out
the :doc:`MBTR tutorial <mbtr>` to understand the basics of the MBTR framework. The main
differences compared to MBTR are:

  - The :math:`k=1` term has been removed. Encoding the species present within
    a local region is quite tricky, and would essentially require some kind of
    distance information to weight their contribution correctly, making the
    :math:`k=1` term more closer to :math:`k=2` term.
  - LMBTR uses the chemical species X (atomic number 0) for the central
    position. This makes it possible to also encode spatial locations that are
    not centered at any particular atom. It does however mean that you should
    be careful not to mix information from outputs that have different
    central species. If the chemical identify of the central species is
    important, you may want to consider a custom output stratification scheme
    based on the chemical identity of the central species. When used as input
    for machine learning, training a separate model for each central species is
    also possible.

Setup
-----

Instantiating an LMBTR descriptor can be done as follows:

.. literalinclude:: ../../../examples/lmbtr.py
   :language: python
   :lines: 1-22

The arguments have the following effect:

.. automethod:: dscribe.descriptors.lmbtr.LMBTR.__init__


Creation
--------
After LMBTR has been set up, it may be used on atomic structures with the
:meth:`~.LMBTR.create`-method.

.. literalinclude:: ../../../examples/lmbtr.py
   :language: python
   :start-after: Create
   :lines: 1-9

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.lmbtr.LMBTR.create

The output will in this case be a numpy array with shape [#positions,
#features]. The number of features may be requested beforehand with the
:meth:`~.LMBTR.get_number_of_features`-method.

Examples
--------
The following examples demonstrate common use cases for the descriptor. These
examples are also available in dscribe/examples/lmbtr.py.

Adsorption site analysis
~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrate the use of LMBTR as a way of analysing local sites in
a structure. We build an Al(111) surface and analyze four different adsorption
sites on this surface: top, bridge, hcp and fcc.

.. literalinclude:: ../../../examples/lmbtr.py
   :language: python
   :start-after: Surface sites
   :lines: 1-13

These four sites are described by LMBTR with pairwise :math:`k=2` term.

.. literalinclude:: ../../../examples/lmbtr.py
   :language: python
   :start-after: Surface sites
   :lines: 16-32

Plotting the output from these sites reveals the different patterns in these
sites.

.. literalinclude:: ../../../examples/lmbtr.py
   :language: python
   :start-after: Surface sites
   :lines: 34-

.. figure:: /_static/img/sites.png
   :width: 640px
   :height: 480px
   :scale: 80 %
   :alt: LMBTR sites
   :align: center

   The LMBTR k=2 fingerprints for different adsoprtion sites on an Al(111)
   surface.

Correctly tuned, such information could for example be used to train an
automatic adsorption site classifier with machine learning.

.. bibliography:: ../references.bib
   :style: unsrt
   :filter: docname in docnames

