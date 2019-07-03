Installation
============
The newest versions of the package are compatible with Python 3.X (tested on
3.5, 3.6 and 3.7). The versions <= 0.2.7 also support Python 2.7. We currently only
support Unix-based systems, including Linux and macOS. For Windows-machines we
suggest using the `Windows Subsystem for Linux (WSL)
<https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux>`_. The exact list
of dependencies are given in setup.py and all of them will be automatically
installed during setup.

The latest stable release is available through pip: (add the *-\\-user* flag if
root access is not available)

.. code-block:: sh

    pip install dscribe

To install the latest development version, clone the source code from
github and install with pip from local file:

.. code-block:: sh

    git clone https://github.com/SINGROUP/dscribe.git
    cd dscribe
    pip install .

Common issues
-------------
 - **fatal error: Python.h: No such file or directory**: The package depends on
   C/C++ extensions that are compiled during the setup. For the compilation to
   work you will need to install the *pythonX.X-dev*-package, where X.X is the
   python version you use to run dscribe. E.g. for python 3.7 on Ubuntu this
   package could be installed with:

   .. code-block:: sh

       sudo apt install python3.7-dev
