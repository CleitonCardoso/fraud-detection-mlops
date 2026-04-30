# CLAUDE.md — Fraud Detection MLOps (FIAP MLET Datathon Phase 05)

## Visão Geral do Projeto

Sistema MLOps completo para detecção de fraude em transações de cartão de crédito.
Dataset: UCI Credit Card Fraud (284.807 transações, 0.17% fraude).
Stack: FastAPI + MLflow + scikit-learn + LangGraph + FAISS + Prometheus + Grafana + Evidently.

## Estrutura de Diretórios

```
src/
  features/         feature engineering (compute_features, feature store)
  models/           baseline.py (LR + RF), mlp.py (PyTorch), train.py (pipeline)
  monitoring/       metrics.py (Prometheus), drift.py (Evidently PSI)
  security/         guardrails.py (input/output), pii_detection.py (Presidio)
  serving/          app.py (FastAPI), Dockerfile
  agent/            react_agent.py (LangGraph), tools.py (3 tools), rag_pipeline.py (FAISS)
evaluation/         ragas_eval.py, llm_judge.py, golden_set_builder.py
tests/              pytest com cobertura >= 60%
configs/            prometheus.yml, monitoring_config.yaml, grafana/
docs/               MODEL_CARD.md, SYSTEM_CARD.md, LGPD_PLAN.md, OWASP_MAPPING.md, adr/
notebooks/          01_eda.ipynb
data/
  raw/              creditcard.csv (rastreado pelo DVC)
  processed/        feature_store.parquet, faiss_index/, drift_report.html
  golden_set/       golden_set.json (20 pares Q&A)
```

## Comandos Essenciais

```bash
make setup          # instala dependências
make data           # baixa dataset via Kaggle
make train          # treina LR + RF + MLP, loga no MLflow
make test           # pytest com cobertura >= 60%
make serve          # inicia FastAPI em localhost:8000
make drift          # roda Evidently e atualiza PSI no Prometheus
make eval           # RAGAS + LLM-judge contra o golden set
make lint           # ruff + mypy + bandit
docker compose up -d  # sobe todos os serviços
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
- Commits locais são livres — **NUNCA fazer push** para o remoto neste repositório
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
