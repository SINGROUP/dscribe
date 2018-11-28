# Development dependencies
pip install -r devrequirements.txt
pip install .

# Compile CMBTR extension with cython. The .so file is not generated, as it
# will be compiled during package setup
cythonize dscribe/libmbtr/cmbtrwrapper.pyx

# TODO: Compile documentation with Sphinx
