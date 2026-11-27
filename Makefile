# Shortcuts for the most common tasks. Run `make help` to list them.
.DEFAULT_GOAL := help
PYTHON ?= python
MODEL ?= custom_cnn

.PHONY: help install install-dev data validate test lint format train tune baseline evaluate api demo docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and runtime dependencies
	$(PYTHON) -m pip install -e .

install-dev: ## Install dev tools (tests, lint, pre-commit) and git hooks
	$(PYTHON) -m pip install -e . -r requirements-dev.txt
	pre-commit install

data: ## Run the full DVC data pipeline (download -> validate)
	dvc repro

validate: ## Re-run only the data validation stage
	dvc repro validate

test: ## Run the unit tests
	$(PYTHON) -m pytest

lint: ## Check code style
	ruff check src tests api
	ruff format --check src tests api

format: ## Auto-format code
	ruff check --fix src tests api
	ruff format src tests api

baseline: ## Majority-class baseline on the test set
	$(PYTHON) -m brain_tumor.baseline

train: ## Train a model, e.g. `make train MODEL=resnet18`
	$(PYTHON) -m brain_tumor.train train.model=$(MODEL)

tune: ## Optuna hyperparameter search, e.g. `make tune MODEL=custom_cnn`
	$(PYTHON) -m brain_tumor.tune train.model=$(MODEL)

evaluate: ## Evaluate a checkpoint, e.g. `make evaluate CKPT=models/custom_cnn/best.pt`
	$(PYTHON) -m brain_tumor.evaluate --checkpoint $(CKPT)

api: ## Start the FastAPI inference server on :8000
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

demo: ## Start the Gradio demo on :7860
	$(PYTHON) -m api.gradio_app

docker: ## Build the inference Docker image
	docker build -t brain-tumor-api .

clean: ## Remove caches (keeps data and models)
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
