# Platform core: test, lint, demo.
#
# Deliberately plain make: it has to work on a developer laptop, a CI runner and
# a Raspberry Pi gateway without installing a build system first.

PY      ?= python3
VENV    ?= .venv
BIN      = $(VENV)/bin

.PHONY: help venv install test test-fast lint demo clean

help:
	@echo "make install    create $(VENV) and install core[ml,field,dev]"
	@echo "make test       run the full test suite (core + edge)"
	@echo "make lint       ruff check"
	@echo "make demo       commissioning -> degradation -> alarm -> RUL walk-through"
	@echo "make clean      remove build artefacts and the venv"

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install -q -e "core[ml,field,dev]"

test: install
	$(BIN)/python -m pytest core/tests edge/tests -q

test-fast: install
	$(BIN)/python -m pytest core/tests edge/tests -q -x --ff

lint: install
	$(BIN)/ruff check core/pdmcore || true

demo: install
	$(BIN)/pdm demo --machine motor_pump_37kw
	$(BIN)/pdm payload

clean:
	rm -rf $(VENV) core/build core/dist core/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +

.PHONY: ci-install
ci-install:
	@mkdir -p .github/workflows
	@cp ci/github/*.yml .github/workflows/
	@echo "copied $$(ls ci/github/*.yml | wc -l) workflow file(s) into .github/workflows/"
	@echo "note: committing these needs a token with the 'workflow' scope -- see ci/README.md"
