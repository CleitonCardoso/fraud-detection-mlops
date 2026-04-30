# Fraud Detection MLOps System

> FIAP Pós-Tech MLET — Datathon Fase 05 | Integrative Project (Phases 01–05)

MLOps system for credit card fraud detection — from data ingestion to LLM-powered agent with full observability, security, and governance.

## Quick start

```bash
cp .env.example .env          # fill in your keys
make setup                    # install dependencies
make data                     # download dataset (requires Kaggle account)
make train                    # train models, log to MLflow
make serve                    # start all services via docker-compose
make test                     # run test suite
```

## Services

| Service | URL | Description |
|---|---|---|
| API | http://localhost:8000/docs | FastAPI + Swagger UI |
| MLflow | http://localhost:5000 | Experiment tracking + Model Registry |
| Grafana | http://localhost:3000 | Monitoring dashboard (admin/admin) |
| Langfuse | http://localhost:3001 | LLM telemetry |
| Prometheus | http://localhost:9090 | Metrics scraping |

## Architecture

See [PLAN.md](PLAN.md) for the full architecture, diagrams, and 4-day implementation plan.
