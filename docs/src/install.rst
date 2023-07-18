Installation
============
The newest versions of the package are compatible with Python >= 3.8 (tested on
3.8, 3.9, 3.9, 3.10 and 3.11). DScribe versions <= 0.2.7 also support Python 2.7.

pip
---
The latest stable release is available through pip:

.. code-block:: sh

    pip install dscribe

Since version 2.0.1, `wheel distributions
<https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#wheels>`_
are available for several platforms using `cibuildwheel
<https://github.com/pypa/cibuildwheel>`_. For more exotic platforms a
source distribution is available, but this will require a compilation step
during installation.

conda
-----
We also provide a conda package through the `conda-forge
<https://conda-forge.org/>`_ project. To install the package with conda, use
the following command:

.. code-block:: sh

   conda install -c conda-forge dscribe

From source
-----------
To install the latest development version from source, clone the source code
from github and install with pip from local file:

.. code-block:: sh

    git clone https://github.com/SINGROUP/dscribe.git
    cd dscribe
    git submodule update --init
    pip install .

When installing from source, the contained C++ extensions are automatically
compiled during install. On Linux-based systems the compilation tools are
typically installed by default, on MacOS you may need to install additional
command line tools if facing issues during setup (see common issues below).

Common issues
-------------
 - **fatal error: pybind11/pybind11.h: No such file or directory**: The package
   depends on pybind11 which is used for communication between python and the
   C++ extensions. Although pybind11 is specified as a requirement in setup.py,
   you may experience issues in pip correctly finding it. In this case you will
   need to install pybind11 before attempting to install dscribe by using the
   following command:

   .. code-block:: sh

       pip install pybind11

 - **fatal error: Python.h: No such file or directory**: The package depends on
   C/C++ extensions that are compiled during the setup. For the compilation to
   work you will need to install the *pythonX.X-dev*-package, where X.X is the
   python version you use to run dscribe. E.g. for python 3.7 on Ubuntu this
   package could be installed with:

   .. code-block:: sh

       sudo apt install python3.7-dev

 - **Installation errors on MacOS**: The package depends on C++ extensions that
   are compiled during the setup. If experiencing problems with setup on MacOS,
   you may need to install the Xcode Command Line tools package. This can be
   done with:

   .. code-block:: sh

       xcode-select --install
