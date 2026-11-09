.PHONY: help venv install fetch run test clean

help:
	@echo "make install   create .venv and install dependencies"
	@echo "make fetch     download the free public datasets (~380 MB)"
	@echo "make run       run the full pipeline"
	@echo "make test      run the unit tests"
	@echo "make clean     remove generated outputs and caches"

venv:
	python3 -m venv .venv

install: venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

fetch:
	PYTHONPATH=src .venv/bin/python -m prospectivity fetch

run:
	PYTHONPATH=src .venv/bin/python -m prospectivity run

test:
	PYTHONPATH=src .venv/bin/python -m pytest

clean:
	rm -rf output/figures output/models output/*.tif output/*.csv output/*.json output/*.html
	rm -rf data/processed/*.npz data/processed/*.csv data/processed/*.tif
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
