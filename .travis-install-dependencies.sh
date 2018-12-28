# Development dependencies
pip install -r devrequirements.txt

# Compile CMBTR extension with cython. The .so file is not generated, as it
# will be compiled during package setup
cythonize dscribe/libmbtr/cmbtrwrapper.pyx

pip install .
