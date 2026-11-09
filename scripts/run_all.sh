#!/usr/bin/env bash
# Reproduce the whole project from an empty checkout.
# Everything downloaded is free and requires no account or API key.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

export PYTHONPATH=src
.venv/bin/python -m prospectivity fetch
.venv/bin/python -m prospectivity run
.venv/bin/python -m pytest
