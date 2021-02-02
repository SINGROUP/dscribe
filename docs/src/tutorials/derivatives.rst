Derivatives
===========

.. note::
   We are incrementally adding support for calculating the derivatives of
   descriptors with respect to the atom positions. In version **0.5.0** you can
   find an initial implementation for getting these derivatives for the SOAP
   descriptor. Also check out the :doc:`new tutorial
   <tutorials/machine_learning/forces_and_energies>` on predicting energies and
   forces using these derivatives.

The descriptor outputs that DScribe provides are typically vectors containing
multiple features. Each of these features has a dependency on the positions of
atoms in the system, and in particular each feature has a derivative with
respect to the x-, y-, and z-components of the position of every atom in the
system.

Some of the descriptors included in DScribe allow you to calculate these
derivatives. The derivatives provide a way to study the effect of different
atoms on the output, and in particular they can be used to build machine
learning based force-fields, see e.g. the tutorial on building one.

We have decided to retain as much structure in the derivative output as
possible, which makes it quite high-dimensional. This approach allows you to
better understand the different components, and you still have the option to
re-arrange the output as you wish before your application. The derivative
output for a single system is organized as follows:

 - For global descriptors the output is three-dimensional: :code:`[n_atoms, 3, n_features]`
 - For local descriptors the output is four-dimensional: :code:`[n_centers, n_atoms, 3, n_features]`

Here the dimension with :code:`n_centers` loops through the different centers
used in a local descriptor, :code:`[n_atoms` loops through the atoms for which
the derivatives were calculated for, the second-to-last dimension with three
components loops through the x, y and z components, and the last dimension with
:code:`[n_features]` loops through the different features. This layout is
convenient for calculating the descriptor values, but depending on your
application, you may need a different layout. You can quite easily rearrange
these dimension with either `np.moveaxis
<https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html>`_ or
`sparse.moveaxis
<https://sparse.pydata.org/en/stable/generated/sparse.moveaxis.html`_, or do
completely custom layouts with some looping.

Notice, that if you calculate the derivatives of several systems in one
function call, one additional dimension is added in front which loops through
the given systems.

If you use the default dense output, the derivatives will be stored in a
regular numpy array. Notice that the size of these dense arrays grows very
quickly with system size. A good way around this is to use sparse arrays
instead by using the sparse=True option in the descriptor constructor. In large
systems the derivative array will typically become quite sparse, giving
significant savings in storage space. Since 0.5.0 We have opted to use the
sparse-library for all of our sparse outputs. Unlike the scipy sparse matrices,
this library can also represent n-dimensional sparse matrices. Whenever you
need a dense or scipy sparse matrices for your algorithms, simply call
todense()/tocsr()/tocsc().

In general, the function call signature for getting the derivatives looks like
this
 - systems: 
 - (positions): For local descriptors. 
 - include: Indices of the atoms agains which the derivatives should be
   calculated for. If no value is specified, all atoms will be used.
 - exclude: 
 - method: In the derivative function call, you sometimes have the option for
   choosing between analytical and numerical derivatives. In general, if
   analytical derivatives ara available (check the function documentation),
   they will be somewhat faster to compute, but some limitations to their
   applicability may apply. Any possible limitations are due to the fact that
   the analytical form could not be identified or has not been implemented as
   of yet. The numerical derivatives are implemented with a centered finite
   different scheme that has a good balance between accuracy and speed.
 - return_descriptor: Whether or not to return the descriptor as well. If you
   anyways need to compute the descriptor, it is typically faster to use this
   option instead of a separate create()-call. 
