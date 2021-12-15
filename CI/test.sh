#!/usr/bin/env bash
cd ../regtests
export COVERAGE_FILE="../.coverage"
coverage run --source="dscribe" testrunner.py
if [ $? -ne 0 ]; then
    exit 1
fi
coverage run -m --source="dscribe" --append pytest
