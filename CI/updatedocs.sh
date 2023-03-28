version=2.0.x

# Build docs, copy to correct docs folder, delete build
cd ../docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html SPHINXOPTS="-D version=$version -D release=$version"
rm -r ../$version
mkdir ../$version
cp -a build/html/. ../$version
rm -r build
cd ..
rm stable
rm latest
ln -s $version stable
ln -s $version latest
