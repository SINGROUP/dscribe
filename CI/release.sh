# Build source distribution (cannot build a wheel due to C/C++ extension)
pip setup.py sdist
# Upload to PyPI
twine upload dist/*
