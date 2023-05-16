Valle-Oganov descriptor
=======================
Implementation of the fingerprint descriptor by Valle and Oganov
:cite:`valle2010crystal` for :math:`k=2` and :math:`k=3`
:cite:`bisbo2020global`. Note that the descriptor is implemented as a subclass
of MBTR and can also can also be accessed in MBTR by setting the right
parameters: see the :doc:`MBTR tutorial <mbtr>`.

It is advisable to first check out the :doc:`MBTR tutorial <mbtr>` to
understand the basics of the MBTR framework. The differences compared to MBTR
are:

  - Only geometry functions :code:`distance` and :code:`angle` are available.
  - A radial cutoff distance :code:`r_cut` is used.
  - In the grid setup, :code:`min` is always 0 and :code:`max` has the same value as :code:`r_cut`.
  - Normalization and weighting are automatically set to their Valle-Oganov options.
  - :code:`periodic` is set to True: Valle-Oganov normalization doesn't support non-periodic systems.
  - Parameters :code:`species` and :code:`sparse` work similarly to MBTR.

Setup
-----
Instantiating a Valle-Oganov descriptor can be done as follows:

.. literalinclude:: ../../../../examples/valleoganov.py
   :language: python
   :lines: 1-11

The arguments have the following effect:

.. automethod:: dscribe.descriptors.valleoganov.ValleOganov.__init__

In some publications, a grid parameter :math:`\Delta`, which signifies the
width of the spacing, is used instead of :code:`n`. However, here :code:`n` is used in order
to keep consistent with MBTR. The correlation between :code:`n` and
:math:`\Delta` is :math:`n=(max-min)/\Delta+1=(r_{cutoff})/\Delta+1`.

Creation
--------
After the descriptor has been set up, it may be used on atomic structures with the
:meth:`~.ValleOganov.create`-method.

.. literalinclude:: ../../../../examples/valleoganov.py
   :language: python
   :start-after: Create
   :lines: 1-21

The call syntax for the create-function is as follows:

.. automethod:: dscribe.descriptors.mbtr.MBTR.create
   :noindex:

The output will in this case be a numpy array with shape :code:`[n_positions, n_features]`.
The number of features may be requested beforehand with the
:meth:`~.ValleOganov.get_number_of_features`-method.

Examples
--------
The following examples demonstrate some use cases for the descriptor. These
examples are also available in dscribe/examples/valleoganov.py.

Visualization
~~~~~~~~~~~~~

The following snippet demonstrates how the output for :math:`k=2` can be
visualized with matplotlib. Visualization works very similarly to MBTR.

.. literalinclude:: ../../../../examples/valleoganov.py
   :start-after: Visualization
   :language: python
   :lines: 1-34

.. figure:: /_static/img/vo_k2.png
   :width: 1144px
   :height: 772px
   :scale: 60 %
   :alt: Valle-Oganov k=2
   :align: center

   The Valle-Oganov output for k=2. The graphs for Na-Na and Cl-Cl overlap due to their
   identical arrangement in the crystal.

Setup with MBTR class
~~~~~~~~~~~~~~~~~~~~~

For comparison, the setup for Valle-Oganov descriptor for the previous structure, but using
the MBTR class, would look like the following.

.. literalinclude:: ../../../../examples/valleoganov.py
   :start-after: MBTR setup for the same structure and descriptor
   :language: python
   :lines: 1-13

Side by side with MBTR output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A graph of the output for the same structure created with different descriptors. Also demonstrates how the :ref:`Valle-Oganov normalization <norm-label>` for k2 term converges at 1.

.. literalinclude:: ../../../../examples/valleoganov.py
   :start-after: Comparing to MBTR output
   :language: python
   :lines: 1-42

.. figure:: /_static/img/vo_mbtr.png
   :width: 1144px
   :height: 772px
   :scale: 60 %
   :alt: Valle-Oganov and MBTR comparison
   :align: center

   Outputs for k=2. For the sake of clarity, the cutoff distance has been lengthened and only the Na-Cl pair has been plotted.

.. bibliography:: ../../references.bib
   :style: unsrt
   :filter: docname in docnames

