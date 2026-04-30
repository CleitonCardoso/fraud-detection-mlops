# System Card — Fraud Detection MLOps System

> Seguindo o padrão de Mitchell et al. (2019) adaptado para sistemas LLM compostos

---

## Descrição do Sistema

Sistema MLOps completo para detecção de fraude em transações financeiras, composto por:
1. Modelo de classificação (Random Forest + MLP PyTorch)
2. Agente LLM ReAct com 3 ferramentas
3. Pipeline RAG sobre base de conhecimento de fraude
4. Stack de observabilidade (Prometheus, Grafana, Langfuse, Evidently)
5. Guardrails de segurança (input/output) com Presidio

## Arquitetura

```
Usuário → InputGuardrail → Agente ReAct → [fraud_predictor | transaction_lookup | drift_report]
                                        ↓
                               OutputGuardrail → Resposta
```

Serving: FastAPI em Google Cloud Run (HTTPS, scale-to-zero)
Modelos: MLflow Registry + GitHub Container Registry
Monitoramento: Prometheus + Grafana Cloud + Langfuse Cloud

## Riscos Sistêmicos

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Alucinação do LLM em resposta sobre fraude | Média | Alto | RAGAS faithfulness, OutputGuardrail, RAG com fontes verificadas |
| Prompt injection via query do usuário | Baixa | Alto | InputGuardrail com 9 padrões, bloqueio com HTTP 400 |
| Drift silencioso do modelo | Média | Alto | Evidently diário, PSI duplo threshold, alerta automático |
| Exposição de PII no output | Baixa | Alto | Presidio no OutputGuardrail, logging sem dados sensíveis |
| Falso negativo (fraude não detectada) | Baixa | Alto | Recall monitorado, threshold ajustável, revisão humana obrigatória |
| Indisponibilidade da OpenAI API | Baixa | Médio | /predict funciona sem o agente; fallback para Ollama local |
| Regressão por deploy de modelo ruim | Baixa | Alto | Champion-challenger + human-in-the-loop approval gate |

## Limites de Uso

- O sistema NÃO toma decisões autônomas de bloqueio — é ferramenta de apoio à decisão humana
- Não deve ser usado como única fonte de verdade em investigações de fraude
- Não processa dados com PII direta — features são PCA-anonimizadas

## Responsáveis

| Papel | Responsabilidade |
|---|---|
| ML Engineer | Treino, deploy, monitoramento, retraining |
| Data Scientist | Feature engineering, análise de drift, fairness |
| Security | Red teaming, OWASP mapping, guardrails |
| Compliance | LGPD, aprovação de produção (human-in-the-loop) |

## Avaliação de Qualidade do Agente

| Métrica | Método | Score alvo |
|---|---|---|
| Faithfulness | RAGAS | ≥ 0.85 |
| Answer Relevancy | RAGAS | ≥ 0.80 |
| Context Precision | RAGAS | ≥ 0.75 |
| Context Recall | RAGAS | ≥ 0.75 |
| Precisão técnica | LLM-as-judge | ≥ 4.0/5.0 |
| Clareza | LLM-as-judge | ≥ 4.0/5.0 |
| Impacto de negócio | LLM-as-judge | ≥ 3.5/5.0 |
