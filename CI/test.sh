#!/usr/bin/env bash
cd ../regtests
export COVERAGE_FILE="../.coverage"
coverage run --source="dscribe" testrunner.py
coverage run -m --source="dscribe" --append pytest
