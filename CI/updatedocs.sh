version=0.5.0rc1

# Build docs, copy to correct docs folder, delete build
cd ../docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html SPHINXOPTS="-D version=$version -D release=$version"
cp -a build/html/. ../latest
rm -r build
