.. _tutorials:

Tutorials
=========

Basics
------

We provide several tutorials that will help you get acquainted with DScribe and
descriptors in general. If you do not have much experience with descriptors, or
you simply want to refresh your memory, the tutorial on :doc:`Basic concepts
<basics>` is a great place to start.

.. toctree::
   :maxdepth: 1

   basics
   derivatives
   sparse

Descriptors
-----------

We have provided a basic tutorial for each descriptor included in DScribe.
These tutorials briefly introduce the descriptor and demonstrate their basic
call signature. We have also included several examples that should cover many
of the use cases.

.. toctree::
   :maxdepth: 1

   descriptors/coulomb_matrix
   descriptors/sine_matrix
   descriptors/ewald_sum_matrix
   descriptors/acsf
   descriptors/soap
   descriptors/mbtr
   descriptors/lmbtr
   descriptors/valleoganov


Machine Learning
----------------

In the machine learning-section you can find few real-life examples that
demonstrate how DScribe can be used in machine learning applications.

.. toctree::
   :maxdepth: 1

   machine_learning/forces_and_energies
   machine_learning/clustering


Visualization
----------------

Mapping the descriptors into a visual representation is useful in analyzing
complex systems. These examples demonstrate how to achieve this with DScribe.

.. toctree::
   :maxdepth: 1

   visualization/coloring_by_environment


Similarity Analysis
-------------------

Measuring the similarity of structures is key to several machine learning
applications --- especially kernel-based methods -- but these techniques are
also very useful in other applications. We provide few basic tutorials on how
the similarity of structures can be quantified using DScribe.

.. toctree::
   :maxdepth: 1

   similarity_analysis/kernels
