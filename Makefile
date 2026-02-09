.PHONY: help install test lint format clean run restore-models train

help:
	@echo "ML Stock Predictor - Available Commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run code quality checks"
	@echo "  make format        - Format code (if tools available)"
	@echo "  make clean         - Clean temporary files"
	@echo "  make restore-models - Restore models from base64"
	@echo "  make train         - Train enhanced model"
	@echo "  make run           - Run Streamlit app"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

lint:
	@echo "Running basic Python syntax checks..."
	python -m py_compile app/*.py src/*.py scripts/*.py tests/*.py

format:
	@echo "Code formatting - install black/isort for auto-formatting"
	@command -v black >/dev/null 2>&1 && black src/ app/ scripts/ tests/ || echo "black not installed"
	@command -v isort >/dev/null 2>&1 && isort src/ app/ scripts/ tests/ || echo "isort not installed"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -f ml_predictor.log

restore-models:
	python scripts/restore_models.py

train:
	python scripts/train_model.py

run:
	streamlit run app/streamlit_app.py
