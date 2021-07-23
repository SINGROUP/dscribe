Basic concepts
==============

This tutorial covers the very basic concepts that are needed to get started
with DScribe and descriptors in general. Please read through this page before
moving on to other tutorials if you have no previous experience or
simply want to refresh some core concepts.

DScribe provides methods to transform atomic structures into fixed-size numeric
vectors. These vectors are built in a way that they efficiently summarize the
contents of the input structure. Such a transformation is very useful for
various purposes, e.g.

 - Input for supervised machine learning models, e.g. regression.
 - Input for unsupervised machine learning models, e.g. clustering.
 - Visualizing and analyzing a local chemical environment.
 - Measuring similarity of structures or local structural sites.
 - etc.

Please read more details in our open-access article: `DScribe: Library of
descriptors for machine learning in materials science
<https://doi.org/10.1016/j.cpc.2019.106949>`_

Terminology
-----------
 - *structure*: An atomic geometry containing the chemical species, atomic
   positions and optionally a unit cell for periodic structures.
 - *descriptor*: A particular method for transforming a structure into a
   constant sized vector. There are various options which are suitable for
   different use cases. DScribe currently provides the following descriptors: :doc:`Coulomb matrix <coulomb_matrix>`, :doc:`Sine matrix <sine_matrix>`, :doc:`Ewald sum matrix <ewald_sum_matrix>`, :doc:`Atom-centered Symmetry Functions (ACSF) <acsf>`, :doc:`Smooth Overlap of Atomic Positions (SOAP) <soap>`, :doc:`Many-body Tensor Representation (MBTR) <mbtr>` and :doc:`Local Many-body Tensor Representation (LMBTR) <lmbtr>`.
 - *descriptor object*: In DScribe there is a single python class for each
   descriptor. The object that is instantiated from this class is called a
   descriptor object.
 - *feature vector*: The descriptor objects produce a single one-dimensional
   vector for each input structure. This is called a feature vector.
 - *feature*: A single channel/dimension in the multi-dimensional feature
   vector produced by a descriptor object for a structure. Each feature is a
   number that represents a specific structural/chemical property in the
   structure.

Typical workflow
----------------
1. DScribe uses the `Atomic Simulation
   Environment (ASE) <https://wiki.fysik.dtu.dk/ase/>`_ to represent and work
   with atomic structures as it provides convenient ways to read, write, create and
   manipulate them. The first step is thus to transform your
   atomic structures into `ASE Atoms
   <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_. 

   For example:

   .. literalinclude:: ../../../examples/basics.py
    :language: python
    :lines: 1-8

2. Usually the descriptors require some knowledge about the dataset you are
   analyzing. This means that you wll need to gather information about the
   expected input space of all your analyzed structures. Often simply gathering
   a list of the present chemical species is enough.

   For example:

   .. literalinclude:: ../../../examples/basics.py
    :language: python
    :lines: 10-15

3. Setup the descriptor object. The exact setup depends on the used descriptor
   and your use case. Notice that typically you will want to instantiate only
   one descriptor object which will handle all structures in your dataset. You
   should read the original articles or the specific tutorials to understand
   the meaning of different settings. For machine learning purposes you may
   also want to cross-validate the different settings to find the
   best-performing ones.

   For example:

   .. literalinclude:: ../../../examples/basics.py
    :language: python
    :lines: 17-28

4. Call the :code:`create()` function of the descriptor object on a single Atoms object
   or a list of them. Optionally provide a number of cores to parallelize the work.

   .. literalinclude:: ../../../examples/basics.py
    :language: python
    :lines: 30-31

5. The output is either 2D (number of structures :math:`\times` number of features) `numpy
   array
   <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_ or
   `scipy sparse matrix
   <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ (depends on the
   :code:`sparse` setting of your descriptor object) that you can store store
   for later use. Have fun!
