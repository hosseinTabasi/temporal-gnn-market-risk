PYTHON ?= python
.PHONY: test toy lint
test:
	$(PYTHON) -m pytest -q
toy:
	PYTHONPATH=src $(PYTHON) -m run_toy
lint:
	$(PYTHON) -m ruff check src tests
