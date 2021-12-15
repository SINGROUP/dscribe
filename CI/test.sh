#!/usr/bin/env bash
cd ../regtests
export COVERAGE_FILE="../.coverage"
unittest=`coverage run --source="dscribe" testrunner.py`
pytest=`coverage run -m --source="dscribe" --append pytest`
if [ "$unittest" != 0 ] || [ "$pytest" != 0 ]; then
    exit 1
fi
