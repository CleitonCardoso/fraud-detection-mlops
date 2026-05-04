.PHONY: setup data train retrain serve test eval drift demo lint format clean localstack-init data-push data-pull

# ── Setup ──────────────────────────────────────────────────────────────
setup:
	pip install -e ".[ml,serving,agent,monitoring,security,storage,dev]"
	pre-commit install
	python -m spacy download en_core_web_sm
	python -m spacy download pt_core_news_sm

# ── Data ───────────────────────────────────────────────────────────────
data:
	@echo "Baixando dataset via Kaggle CLI..."
	kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
	dvc add data/raw/creditcard.csv
	@echo "Dataset pronto em data/raw/creditcard.csv"

# ── LocalStack S3 ──────────────────────────────────────────────────────
localstack-init:
	@echo "Aguardando LocalStack ficar pronto..."
	@until curl -sf http://localhost:4566/_localstack/health | python3 -c "import sys,json; h=json.load(sys.stdin); sys.exit(0 if h.get('services',{}).get('s3')=='available' else 1)" 2>/dev/null; do sleep 2; done
	@echo "Criando bucket S3 no LocalStack..."
	awslocal s3 mb s3://fraud-detection-features --region us-east-1 2>/dev/null || echo "Bucket já existe"
	awslocal s3 ls
	@echo "LocalStack S3 pronto: s3://fraud-detection-features"

data-push:
	@echo "Versionando feature store e enviando para LocalStack S3..."
	dvc repro features
	AWS_REQUEST_CHECKSUM_CALCULATION=when_required dvc push
	@echo "Feature store enviado para s3://fraud-detection-features"

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

retrain:
	@echo "Verificando drift antes de retreinar..."
	@python src/monitoring/drift.py; \
	if [ $$? -ne 0 ]; then \
		echo "Drift detectado — iniciando retreinamento..."; \
		python src/models/train.py; \
	else \
		echo "Sem drift crítico — retreinamento não necessário."; \
	fi

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
	make localstack-init
	python src/models/train.py
	make data-push
	python evaluation/ragas_eval.py
	@echo ""
	@echo "Sistema pronto:"
	@echo "  API:        http://localhost:8000/docs"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Grafana:    http://localhost:3000"
	@echo "  Langfuse:   http://localhost:3001"
	@echo "  LocalStack: http://localhost:4566"

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf htmlcov/ .coverage coverage.xml
