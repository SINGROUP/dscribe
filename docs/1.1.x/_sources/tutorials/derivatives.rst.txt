Derivatives
===========
.. note::
   Currently, derivatives are only available for the SOAP descriptor. We will
   incrementally add support for getting the derivatives of other descriptors
   as well.

The descriptor outputs that DScribe provides are typically vectors containing
multiple features. Each of these features has a dependency on the positions of
atoms in the system, and in particular each feature has a derivative with
respect to the x-, y-, and z-components of the position of every atom in the
system.

Some of the descriptors included in DScribe allow you to calculate these
derivatives. The derivatives provide a way to study the effect of different
atoms on the output, and in particular they can be used to build machine
learning based force-fields, see :doc:`the tutorial on building one
<machine_learning/forces_and_energies>`.

Call signature
--------------
The descriptors for which derivatives have been made available have a new
:code:`derivatives`-function, see e.g. :meth:`.SOAP.derivatives`. This function works
very similarly to the create function and the typical function call signature
looks like this:

 - :code:`systems`: one or multiple atomic systems
 - :code:`positions`: (only for local descriptors) the positions at which the
   descriptor is evaluated.
 - :code:`include`: Indices of the atoms which should be included in the
   derivative calculations. If no value is specified, all atoms will be used.
 - :code:`exclude`: Indices of atoms which should be excluded from the
   derivative calculations. Use either :code:`include` or :code:`exclude`, not
   both.
 - :code:`method`: Possible values are: :code:`analytical`, :code:`numerical`
   or :code:`auto`. Defaults to :code:`auto`, which means that the most
   efficient available method is used. In general, analytical derivatives are
   preferred, since they are faster to compute. However, they are not always
   available/implemented (check the descriptor documentation). The numerical
   derivatives are implemented with a centered finite difference scheme that
   has a good balance between accuracy and speed.
 - :code:`attach`: (only for local descriptors) controls the behaviour of positions
   defined as atomic indices. If True, the positions tied to an atomic index
   will move together with the atoms with respect to which the derivatives are
   calculated against. If False, positions defined as atomic indices will be
   converted into cartesian locations that are completely independent of the
   atom location during derivative calculation.
 - :code:`return_descriptor`: Whether or not to return the descriptor as well.
   If you anyways need to compute the descriptor, it is typically faster to use
   this option instead of a separate :code:`create`-call. 
 - :code:`n_jobs`: Number of parallel jobs. Only applicable when multiple
   systems are provided, upon which the job is split between multiple processes.

Layout
------
We have decided to retain as much structure in the derivative output as
possible. This approach allows you to better understand the different
components, and you still have the option to re-arrange the output as you wish.
The derivative output for a single system is organized as follows:

 - For global descriptors the output is three-dimensional: :code:`[n_atoms, 3, n_features]`
 - For local descriptors the output is four-dimensional: :code:`[n_centers, n_atoms, 3, n_features]`

Here the dimension with :code:`n_centers` loops through the different centers
used in a local descriptor, :code:`n_atoms` loops through the atoms for which
the derivatives were calculated for, the second-to-last dimension with three
components loops through the x, y and z components, and the last dimension with
:code:`n_features` loops through the different features. This layout is
convenient for calculating the descriptor values, but depending on your
application, you may need a different layout. You can quite easily rearrange
these dimension with either `np.moveaxis <https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html>`_ or
`sparse.moveaxis <https://sparse.pydata.org/en/stable/generated/sparse.moveaxis.html>`_, or do
completely custom layouts with some looping.

.. note::
    Whenever multiple systems are provided, an additional dimension is added
    that runs across the different systems: for systems with the same number of
    atoms the output becomes a five-dimensional array, otherwise the output
    becomes a list of four-dimensional arrays.

If you use the default dense output, the derivatives will be stored in a
regular numpy array. Notice that the size of these dense arrays grows very
quickly with system size. A good way around this is to use sparse arrays
instead by using the :code:`sparse=True` option in the descriptor constructor.
In large systems the derivative array will typically become quite sparse,
giving significant savings in storage space. Since 1.0.0 we have opted to use
the `sparse <https://sparse.pydata.org/en/stable/>`_-library for all of
our sparse outputs, see the :doc:`documentation page on sparse output
<machine_learning/forces_and_energies>`.
