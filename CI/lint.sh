#!/usr/bin/env bash
cd ..
black dscribe --check
black regtests --check
