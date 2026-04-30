# ADR-001 — Random Forest como modelo champion em produção

**Status:** Aceito  
**Data:** 2026-04-29  
**Autores:** Cleiton Cardoso

---

## Contexto

O pipeline treina três modelos: Logistic Regression, Random Forest e MLP PyTorch. Precisamos escolher qual vai para produção como champion.

## Decisão

Escolhemos o **Random Forest** como modelo de produção (`@Production` alias no MLflow Registry).

## Justificativa

| Critério | Logistic Regression | Random Forest | MLP PyTorch |
|---|---|---|---|
| AUC | 0.9733 | 0.9529 | N/A* |
| F1 | 0.1055 | **0.8391** | N/A* |
| Precision | 0.0560 | **0.9605** | N/A* |
| Recall | 0.9082 | 0.7449 | N/A* |
| Explicabilidade (SHAP) | Sim | **Sim** | Não |
| Interpretabilidade | Alta | Alta | Baixa |

*MLP não treinado por incompatibilidade de ambiente (Python 3.13 sem suporte PyTorch).

A LR tem AUC mais alto, mas F1 de 0.10 e precision de 5.6% são inaceitáveis em produção — significaria que 94% das transações bloqueadas seriam legítimas, gerando enorme fricção para o cliente. O RF equilibra corretamente precision e recall para o contexto de negócio.

## Consequências

- **Positivo:** Alta precision (96%) minimiza falsos alarmes e fricção com o cliente.
- **Positivo:** SHAP nativo via TreeExplainer para explicabilidade das predições.
- **Negativo:** Recall de 74.5% — aproximadamente 1 em 4 fraudes passa. Aceitável dado o custo de falsos positivos.
- **Mitigação:** Threshold ajustável de 0.5 para contextos de maior tolerância a falso negativo.
