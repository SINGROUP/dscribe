#!/usr/bin/env bash
cd ../regtests
export COVERAGE_FILE="../.coverage"
coverage run --source="../dscribe" testrunner.py
unittest=$?
coverage run -m --source="../dscribe" --append pytest
pytest=$?
if [ "$unittest" != 0 ] || [ "$pytest" != 0 ]; then
    exit 1
fi
