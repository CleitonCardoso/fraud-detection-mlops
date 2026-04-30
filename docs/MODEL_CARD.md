# Model Card — Fraud Detector RF

> Seguindo o padrão de Mitchell et al. (2019) — _Model Cards for Model Reporting_

---

## Descrição do Modelo

| Atributo | Valor |
|---|---|
| **Nome** | fraud_detector_rf |
| **Versão** | 1.0.0 |
| **Tipo** | Classificação binária (fraude / legítima) |
| **Algoritmo** | Random Forest (scikit-learn) |
| **Framework** | scikit-learn 1.4+ |
| **Data de treino** | 2026-04-29 |
| **Nível de risco** | Alto — decisão com impacto financeiro direto |

## Dados de Treinamento

| Atributo | Valor |
|---|---|
| **Dataset** | Credit Card Fraud Detection (ULB / Kaggle) |
| **Tamanho** | 284.807 transações |
| **Período** | Setembro de 2013 (2 dias de transações europeias) |
| **Features** | V1–V28 (PCA anônimo), Time, Amount → engineered: Amount_scaled, Time_scaled, Hour, Amount_log |
| **Target** | Class (0 = legítima, 1 = fraude) |
| **Desbalanceamento** | 99,83% legítimas / 0,17% fraudes (492 fraudes em 284.807) |
| **Versão DVC** | Rastreada via hash `.dvc` — ver MLflow tag `training_data_version` |

## Métricas de Avaliação (holdout 20%)

| Métrica | Valor |
|---|---|
| **AUC-ROC** | ≥ 0.95 |
| **Precision** | ≥ 0.85 |
| **Recall** | ≥ 0.80 |
| **F1-score** | ≥ 0.82 |

> Métricas exatas registradas no MLflow Registry na run de treinamento.

## Uso Pretendido

**Casos de uso adequados:**
- Apoio à decisão de analistas de fraude em revisão de transações suspeitas
- Triagem automática de transações para investigação humana
- Demonstração de arquitetura MLOps para fins educacionais

**Casos de uso inadequados:**
- Decisão autônoma de bloqueio sem revisão humana
- Aplicação em dados de regiões geográficas diferentes (modelo treinado em dados europeus)
- Inferência em tempo real de altíssimo volume sem monitoramento de drift

## Limitações Conhecidas

- Dataset limitado a 2 dias de setembro de 2013 — padrões de fraude modernos podem diferir
- Features são componentes PCA — não é possível interpretar V1-V28 diretamente
- Modelo treinado em transações europeias — pode ter performance reduzida em outros mercados
- Threshold padrão de 0.5 pode precisar de ajuste dependendo do custo relativo de falsos positivos vs. negativos

## Análise de Viés e Fairness

- **Segmentação por valor**: modelo analisado em quintis de Amount — performance estável entre R$0-50, R$50-200, R$200+
- **Segmentação por horário**: performance analisada em turnos (00h-06h, 06h-12h, 12h-18h, 18h-24h)
- **Ferramenta**: fairlearn — ver relatório em `data/processed/fairness_report.json`
- **Conclusão**: nenhum viés sistemático detectado por segmento de valor ou horário

## Explicabilidade

- SHAP TreeExplainer disponível via endpoint `/predict` — retorna top 5 features contribuidoras por predição
- Features mais relevantes: V14, V17, V12, Amount_scaled, V10

## Governança

| Campo | Valor |
|---|---|
| **Responsável** | Ver tag `owner` no MLflow Registry |
| **Aprovação de produção** | Human-in-the-loop gate via GitHub Environment |
| **Retraining** | Semanal (cron) ou quando PSI > 0.2 |
| **Monitoramento** | Evidently (drift) + Grafana (operacional) + Langfuse (LLM) |
| **Conformidade LGPD** | Ver `docs/LGPD_PLAN.md` |
