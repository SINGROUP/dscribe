Chemical Environment Visualization with Local Descriptors
=========================================================

This tutorial demonstrates how one can visualize the abstract descriptor feature
space by mapping it into a visual property. The goal is to give a rough idea on
how to achieve this in a fairly easy scenario where the interesting structural
parts are already known and the system does not contain very many different
phases of interest. In principle the same technique can however be applied to
much more complicated scenarios by using clustering to automatically determine
the reference structures and by mapping reference landmarks in the descriptor
space into any combination of different visual properties, e.g. fill color,
opacity, outline color, size etc.

References and final system
---------------------------
We start out by creating bulk systems for two different iron phases: one for
BCC and another for FCC:

.. literalinclude:: ../../../../examples/visualization/visualization.py
    :language: python
    :lines: 1-15


Since we know the phases we are looking for a priori, we can simply calculate
the reference descriptors for these two environments directly. Here we are using
the LMBTR descriptor, but any local descriptor would do:

.. literalinclude:: ../../../../examples/visualization/visualization.py
    :language: python
    :lines: 17-28

In the more interesting case where the phases are not known in advance, one
could perform some form of clustering (see the :doc:`tutorial on clustering <../machine_learning/clustering>`) to
automatically find out meaningful reference structures from the data itself.
Next we create a larger system that contains these two grains with a bit of
added noise in the atom positions:

.. literalinclude:: ../../../../examples/visualization/visualization.py
    :language: python
    :lines: 30-38

The full structure will look like this:

.. figure:: /_static/img/original.png
   :alt: FCC(111) surface
   :align: center
   :width: 50%

Coloring
--------
Next we want to generate a simple metric that measures how similar the
environment of each atom is to the reference FCC and BCC structures. In this
example we define the metric as the Euclidean distance that is scaled to be
between 0 and 1 from least similar to identical with reference:

.. literalinclude:: ../../../../examples/visualization/visualization.py
    :language: python
    :lines: 40-46

Then we calculate the metrics for all atoms in the system:

.. literalinclude:: ../../../../examples/visualization/visualization.py
    :language: python
    :lines: 48-51

The last step is to create a figure where our custom metric is mapped into
colors. We can create an image where the BCC-metric is tied to the blue color,
and FCC is tied to red:

.. literalinclude:: ../../../../examples/visualization/visualization.py
    :language: python
    :lines: 53-66

The final re-colored system looks like this:

.. figure:: /_static/img/colored.png
   :alt: FCC(111) surface
   :align: center
   :width: 50%

In addition to being able to clearly separate the two grains, one can see
imperfections as noise in the coloring and also how the interfaces affect the
metric both at the FCC/BCC interface and at the vacuum interface.