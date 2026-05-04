# CLAUDE.md — Fraud Detection MLOps (FIAP MLET Datathon Phase 05)

## Visão Geral do Projeto

Sistema MLOps completo para detecção de fraude em transações de cartão de crédito.
Dataset: UCI Credit Card Fraud (284.807 transações, 0.17% fraude).
Stack: FastAPI + MLflow + scikit-learn + LangGraph + FAISS + Prometheus + Grafana + Evidently + LocalStack (S3).

## Estrutura de Diretórios

```
src/
  features/         feature engineering (compute_features, feature store)
  models/           baseline.py (LR + RF), mlp.py (PyTorch), train.py (pipeline)
  monitoring/       metrics.py (Prometheus), drift.py (Evidently PSI)
  security/         guardrails.py (input/output), pii_detection.py (Presidio)
  serving/          app.py (FastAPI), Dockerfile
  agent/            react_agent.py (LangGraph), tools.py (3 tools), rag_pipeline.py (FAISS)
evaluation/         ragas_eval.py, llm_judge.py, golden_set_builder.py, benchmark_configs.py
tests/              pytest com cobertura >= 60%
configs/            prometheus.yml, monitoring_config.yaml, grafana/
docs/               MODEL_CARD.md, SYSTEM_CARD.md, LGPD_PLAN.md, OWASP_MAPPING.md, ARCHITECTURE.md, adr/
notebooks/          01_eda.ipynb
scripts/            build_features.py (DVC pipeline entry point)
data/
  raw/              creditcard.csv (rastreado pelo DVC)
  processed/        feature_store.parquet, features.parquet, faiss_index/, drift_report.html
  golden_set/       golden_set.json (20 pares Q&A)
.dvc/config         DVC remote → LocalStack S3 (s3://fraud-detection-features)
dvc.yaml            Pipeline stages: features → train → drift
dvc.lock            Hashes de todas as saídas do pipeline (commitado no git)
```

## Setup numa Máquina Nova

```bash
# 1. Clonar
git clone https://github.com/CleitonCardoso/fraud-detection-mlops.git
cd fraud-detection-mlops

# 2. Python (usar 3.11 ou 3.12 — PyTorch não suporta 3.13)
python3.12 -m venv .venv && source .venv/bin/activate

# 3. Dependências
pip install -e ".[ml,serving,agent,monitoring,security,dev]"

# 4. Copiar e preencher variáveis de ambiente
cp .env.example .env   # editar com suas keys

# 5. Dataset (requer conta Kaggle com token em ~/.kaggle/kaggle.json)
make data              # baixa creditcard.csv via DVC/Kaggle

# 6. Treinar modelo e registrar no MLflow
mlflow server --host 0.0.0.0 --port 5000 &   # rodar MLflow localmente
make train
# Após o treino, promover RF para produção no MLflow UI (localhost:5000)
# Models → fraud_detector_rf → Aliases → adicionar "Production"

# 7. Subir infraestrutura
docker compose up -d   # Prometheus, Grafana, Langfuse, LocalStack
make localstack-init   # cria bucket S3 no LocalStack (necessário após cada docker compose down -v)

# 7b. Versionar feature store no S3 local (opcional — pode regenerar com make train)
make data-push         # dvc repro features + dvc push → LocalStack S3

# 8. Iniciar API (com MLflow local)
MLFLOW_TRACKING_URI=http://localhost:5000 PYTHONPATH=. \
  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# 9. Ollama (para o agente e avaliações sem OpenAI)
brew install ollama    # macOS
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama serve &

# 10. Buildar FAISS index
PYTHONPATH=. python3 -c "from src.agent.rag_pipeline import build_index; build_index()"
```

## Estado Atual (última sessão)

- **Modelos treinados:** RF (`@Production`) e LR registrados no MLflow local (`mlruns/`)
- **Dados:** `data/raw/creditcard.csv` rastreado pelo DVC (284.807 linhas)
- **Feature store:** `data/processed/feature_store.parquet` gerado e versionado no LocalStack S3
- **FAISS index:** `data/processed/faiss_index/` construído com nomic-embed-text
- **Drift report:** `data/processed/drift_report.html` gerado com Evidently
- **Testes:** 42 passando, 1 skip (torch), cobertura 70.7%
- **Grafana:** provisioned com 11 painéis, 4 alertas (admin/datathon)
- **Ollama:** `llama3.2:3b` + `nomic-embed-text` instalados localmente
- **LocalStack:** rodando como serviço Docker, bucket `fraud-detection-features` criado, feature store versionado via DVC

> ⚠️ `mlruns/` não está no git — numa máquina nova é preciso rodar `make train` novamente e promover o alias `@Production` no MLflow UI.
>
> ⚠️ **LocalStack é ephemeral por padrão.** O volume Docker (`localstack_data`) persiste entre `docker compose stop/up`, mas é apagado com `docker compose down -v`. Numa máquina nova, rodar `make localstack-init` recria o bucket e `make data-push` re-sobe os artefatos. Alternativa: `make train` sempre regenera o feature store do zero a partir do CSV estático.

## Comandos Essenciais

```bash
make setup            # instala dependências
make data             # baixa dataset via Kaggle
make train            # treina LR + RF + MLP, loga no MLflow
make test             # pytest com cobertura >= 60%
make serve            # inicia FastAPI em localhost:8000
make drift            # roda Evidently e atualiza PSI no Prometheus
make eval             # RAGAS + LLM-judge contra o golden set
make lint             # ruff + mypy + bandit
docker compose up -d  # sobe todos os serviços (incl. LocalStack)
make localstack-init  # cria bucket S3 no LocalStack
make data-push        # dvc repro features + dvc push → LocalStack S3
make data-pull        # dvc pull ← LocalStack S3
```

## Serviços e Portas

| Serviço | URL | Credenciais |
|---|---|---|
| FastAPI | http://localhost:8000 | — |
| FastAPI docs | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / datathon |
| Langfuse | http://localhost:3001 | — |
| Ollama | http://localhost:11434 | — |
| LocalStack (S3) | http://localhost:4566 | key=test / secret=test |

## Modelos em Produção

- **Champion:** `fraud_detector_rf@Production` (RF, AUC=0.9529, F1=0.8391, Precision=0.9605)
- **Baseline registrado:** `fraud_detector_lr` (LR, AUC=0.9733, F1=0.1055 — descartado por precision inaceitável)
- Threshold de decisão: 0.5 (ajustável)

## LLM Backend

O agente e as avaliações detectam automaticamente o backend disponível:
1. **Ollama local** (prioridade) — modelos: `llama3.2:3b` (chat), `nomic-embed-text` (embeddings)
2. **OpenAI** (fallback) — configurar `OPENAI_API_KEY` no `.env`

Variáveis de ambiente relevantes:
- `OLLAMA_BASE_URL` — padrão: `http://localhost:11434`
- `OLLAMA_MODEL` — padrão: `llama3.2:1b`
- `OPENAI_API_KEY` — opcional, sobrescreve Ollama se definido

## Variáveis de Ambiente (.env)

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fraud-detection
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PUBLIC_KEY=          # opcional
LANGFUSE_SECRET_KEY=          # opcional
OPENAI_API_KEY=               # opcional
OWNER_EMAIL=your@email.com
```

## Regras de Desenvolvimento

### Código
- Sem comentários óbvios — nomes de variáveis já explicam o quê, comentários só para o porquê não-óbvio
- Sem abstrações prematuras — três linhas similares é melhor que uma abstração prematura
- Type hints obrigatórios em todas as funções públicas
- Sem tratamento de erros para casos impossíveis — não validar o que o framework já garante

### Testes
- `PYTHONPATH=. .venv/bin/pytest tests/ --cov=src --cov-fail-under=60`
- Usar `pytest.importorskip()` para dependências opcionais (torch, langchain)
- Mocks apenas em boundaries externas (MLflow, HTTP) — nunca mockar lógica de negócio

### Git
- Commits locais são livres; push para `https://github.com/CleitonCardoso/fraud-detection-mlops` é permitido
- Mensagens de commit em inglês, seguindo conventional commits

### MLflow
- Usar alias `@Production` (não stages) — MLflow 3.x deprecou stages
- Tags obrigatórias: `model_name`, `model_version`, `model_type`, `training_data_version`, `owner`, `risk_level`, `fairness_checked`, `git_sha`

### Segurança
- Presidio configurado com `language="en"` + recognizers customizados para BR (CPF, CNPJ, RG)
- Guardrails bloqueiam prompt injection com HTTP 400
- OWASP mapping documentado em `docs/OWASP_MAPPING.md`

## Drift e Retreino

- PSI calculado via `src/monitoring/drift.py` com Evidently 0.7
- Threshold warning: PSI > 0.1 → alerta no Grafana
- Threshold retrain: PSI > 0.2 → trigger de retreino automático
- Features monitoradas: `Amount_scaled`, `V14`, `Hour`, `Amount_log1p`

## Compatibilidade

- Python: 3.13 (PyTorch **não** disponível nesta versão — MLP ignorado no treino)
- Evidently: 0.7.x (API diferente de 0.4.x — usar `Dataset.from_pandas` + `DataDefinition`)
- MLflow: 3.11.1 (usar aliases, não stages)
- LangChain: 1.x (usar `langchain_core`, `langchain_text_splitters`, `langgraph`)
