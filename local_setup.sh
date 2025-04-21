#!/usr/bin/env bash

set -Eeuo pipefail

if [ ! -d .venv ]; then
	python -m venv .venv
fi

source .venv/bin/activate

pip install -r requirements.txt

pip install nbdime
nbdime config-git --enable  # enables diff and merge drivers for Git

python download_dataset.py
