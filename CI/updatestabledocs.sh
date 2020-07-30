version=0.4.x

# Build docs, copy to correct docs folder, delete build
cd ../docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html SPHINXOPTS="-D version=$version -D release=$version"
cp -a build/html/. ../latest
rm -r build
