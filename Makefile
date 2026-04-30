.PHONY: setup data train serve test eval drift demo lint format clean

# ── Setup ──────────────────────────────────────────────────────────────
setup:
	pip install -e ".[ml,serving,agent,monitoring,security,dev]"
	pre-commit install
	python -m spacy download en_core_web_sm
	python -m spacy download pt_core_news_sm

# ── Data ───────────────────────────────────────────────────────────────
data:
	@echo "Baixando dataset via Kaggle CLI..."
	kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
	dvc add data/raw/creditcard.csv
	@echo "Dataset pronto em data/raw/creditcard.csv"

data-pull:
	dvc pull

# ── Training ───────────────────────────────────────────────────────────
train:
	python src/models/train.py

# ── Serving ────────────────────────────────────────────────────────────
serve:
	docker compose up --build

serve-dev:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

# ── Testing ────────────────────────────────────────────────────────────
test:
	pytest tests/ -x

test-verbose:
	pytest tests/ -x -v

# ── Evaluation ─────────────────────────────────────────────────────────
eval:
	python evaluation/ragas_eval.py
	python evaluation/llm_judge.py

# ── Monitoring ─────────────────────────────────────────────────────────
drift:
	python src/monitoring/drift.py

# ── Code quality ───────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ evaluation/
	mypy src/ --ignore-missing-imports
	bandit -r src/ -c pyproject.toml

format:
	ruff format src/ tests/ evaluation/
	ruff check src/ tests/ evaluation/ --fix

# ── Demo Day ───────────────────────────────────────────────────────────
demo:
	@echo "Iniciando ambiente completo para Demo Day..."
	docker compose up -d
	@echo "Aguardando serviços..."
	sleep 10
	python src/models/train.py
	python evaluation/ragas_eval.py
	@echo ""
	@echo "Sistema pronto:"
	@echo "  API:      http://localhost:8000/docs"
	@echo "  MLflow:   http://localhost:5000"
	@echo "  Grafana:  http://localhost:3000"
	@echo "  Langfuse: https://cloud.langfuse.com"

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf htmlcov/ .coverage coverage.xml
