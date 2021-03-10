Sparse output
=============

Many of the descriptors become very sparse when using a large chemical space,
or when calculating derivatives. Because of this, DScribe provides the
possibility of creating the output in a sparse format. The sparse output simply
means that only non-zero entries are stored. This can create significant
savings in both RAM and disk usage.

From version 1.0.x onwards, the sparse output uses the :code:`sparse.COO` class
from the `sparse library <https://sparse.pydata.org/en/stable/>`_. The main
benefit compared to e.g. the sparse formats provided by scipy is that
:code:`sparse.COO` supports n-dimensional sparse output with a convenient
slicing syntax.

Persistence
-----------
In order to save/load the sparse output you will need to use the `sparse.save_npz
<https://sparse.pydata.org/en/stable/generated/sparse.save_npz.html>`_/`sparse.load_npz
<https://sparse.pydata.org/en/stable/generated/sparse.load_npz.html>`_
functions from the sparse library. The following example demonstrates this:

.. literalinclude:: ../../../examples/sparse_output.py
   :language: python
   :lines: 1-21

.. note::
    Do not confuse :code:`sparse.save_npz`/:code:`sparse.load_npz` with the
    similarly named functions in :code:`scipy.sparse`.

Conversion
----------
Many external libraries still only support either dense numpy arrays or the 2D
sparse matrices from :code:`scipy.sparse`. This is mostly due to the efficient
linear algebra routines that are implemented for them. Whenever you need such
format, you can simply convert the output provided by DScribe to the needed
format with `todense()
<https://sparse.pydata.org/en/stable/generated/sparse.COO.todense.html#sparse.COO.todense>`_,
`tocsr()
<https://sparse.pydata.org/en/stable/generated/sparse.COO.tocsr.html#sparse.COO.tocsr>`_
or `tocsc()
<https://sparse.pydata.org/en/stable/generated/sparse.COO.tocsc.html#sparse.COO.tocsc>`_:

.. literalinclude:: ../../../examples/sparse_output.py
   :language: python
   :start-after: Convert

.. note::
    Because :code:`scipy.sparse` only suppports 2D sparse arrays, you can only
    call the :code:`tocsr()`/:code:`tocsc()`-functions on 2D slices.
