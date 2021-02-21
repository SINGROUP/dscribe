# Make a source distribution
cd ..
python setup.py sdist

# Install the package itself from the newly created source distribution (this
# way the distribution validity is also checked)
version=`python setup.py --version`
cd dist
tar xvzf dscribe-$version.tar.gz
cd dscribe-$version
pip install .
