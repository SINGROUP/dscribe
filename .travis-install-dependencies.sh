# Development dependencies
pip install -r devrequirements.txt
pip install .

# Compile CMBTR extension with cython. The .so file is not generated, as it
# will be compiled during package setup
cythonize describe/libmbtr/cmbtrwrapper.pyx
