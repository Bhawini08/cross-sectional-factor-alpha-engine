.PHONY: test demo live
export PYTHONPATH=src
test:
	pytest -q
demo:
	python scripts/run_research.py --mode synthetic
live:
	python scripts/run_research.py --mode live
