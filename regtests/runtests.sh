#!/usr/bin/env bash
cd regtests
export COVERAGE_FILE="../.coverage"
coverage run --source="describe" testrunner.py
