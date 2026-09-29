.PHONY: install test check run reproduce research

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

check:
	python -m compileall -q .
	python -m pytest -q

test:
	python -m pytest -q

run:
	python run.py --server

reproduce:
	python run.py --reproduce

research:
	python -m experiments.run_research
