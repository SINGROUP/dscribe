# Build the development docs
./builddevdocs.sh

# Push changes to docs
git config --global user.email "travis@travis-ci.org"
git config --global user.name "Travis CI"
cd ..
git add ./docs
git commit -m "[skip travis] Travis documentation build: $TRAVIS_BUILD_NUMBER"
git push --quiet https://SINGROUP:$GH_TOKEN@github.com/SINGROUP/dscribe master &>/dev/null
