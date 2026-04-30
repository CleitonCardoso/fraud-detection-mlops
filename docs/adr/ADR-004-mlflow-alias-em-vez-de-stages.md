# ADR-004 — MLflow aliases (@Production) em vez de stages

**Status:** Aceito  
**Data:** 2026-04-29  
**Autores:** Cleiton Cardoso

---

## Contexto

MLflow 3.x deprecou o conceito de "stages" (`Staging`, `Production`, `Archived`) do Model Registry em favor de **aliases** flexíveis. O código legado usava `models:/fraud_detector_rf/Production`.

## Decisão

Usar a sintaxe de **alias** do MLflow 3.x: `models:/fraud_detector_rf@Production`.

## Justificativa

- MLflow 3.x remove o endpoint `/api/2.0/mlflow/logged-models` usado pelos stages — usar stages com cliente 3.x gera `404`.
- Aliases são mais flexíveis: um modelo pode ter múltiplos aliases (`@Production`, `@Champion`, `@Canary`).
- Aliases facilitam o fluxo champion-challenger: `@Challenger` e `@Production` coexistem sem conflito.
- Alinhamento com a versão de imagem Docker `ghcr.io/mlflow/mlflow:v3.11.1`.

## Consequências

- **Positivo:** Sem erros 404 em runtime ao carregar o modelo.
- **Positivo:** Suporte nativo a múltiplos aliases por modelo para estratégias de deploy avançadas.
- **Negativo:** Requer promoção manual do alias após treino (`mlflow.set_registered_model_alias`).
- **Negativo:** Aliases não têm transições automáticas — o pipeline de CI/CD precisa gerenciá-los explicitamente.
