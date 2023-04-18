#!/usr/bin/env bash
cd ../tests
export COVERAGE_FILE="../.coverage"
coverage run -m --source="dscribe" pytest
