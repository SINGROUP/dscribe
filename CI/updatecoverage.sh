# Generate coverage report. The environment variable COVERALLS_REPO_TOKEN needs
# to be defined for sending the report to coveralls.
cd ..
coveralls --service=coveralls-python
