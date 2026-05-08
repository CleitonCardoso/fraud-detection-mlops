---
marp: true
theme: default
paginate: true
style: |
  section {
    background: #ffffff;
    color: #1a1a2e;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
    padding: 50px 60px;
  }

  h1 {
    color: #1565c0;
    font-size: 1.8em;
    border-bottom: 3px solid #1565c0;
    padding-bottom: 10px;
    margin-bottom: 20px;
  }

  h2 {
    color: #0d47a1;
    font-size: 1.3em;
  }

  h3 {
    color: #1565c0;
    font-size: 1em;
    margin-bottom: 6px;
  }

  strong {
    color: #1565c0;
  }

  em {
    color: #c62828;
    font-style: normal;
    font-weight: bold;
  }

  code {
    background: #e3f2fd;
    color: #0d47a1;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.85em;
  }

  pre {
    background: #f5f5f5;
    border-left: 4px solid #1565c0;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.75em;
  }

  pre code {
    background: transparent;
    color: #1a1a2e;
    padding: 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
    margin-top: 12px;
  }

  th {
    background: #1565c0;
    color: #ffffff;
    padding: 8px 12px;
    text-align: left;
    border: 1px solid #1565c0;
  }

  td {
    padding: 7px 12px;
    border: 1px solid #bbdefb;
    color: #1a1a2e;
  }

  tr:nth-child(even) td {
    background: #e3f2fd;
  }

  ul, ol {
    line-height: 1.8;
    padding-left: 22px;
  }

  li {
    margin-bottom: 2px;
  }

  blockquote {
    border-left: 4px solid #42a5f5;
    background: #e3f2fd;
    padding: 10px 16px;
    margin: 12px 0;
    color: #0d47a1;
    font-size: 0.88em;
    border-radius: 0 6px 6px 0;
  }

  section.cover {
    background: #1565c0;
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  section.cover h1 {
    color: #ffffff;
    border-bottom: 2px solid rgba(255,255,255,0.4);
    font-size: 2.2em;
  }

  section.cover h2 {
    color: rgba(255,255,255,0.85);
    font-size: 1.2em;
    font-weight: normal;
    margin-top: 8px;
  }

  section.cover p {
    color: rgba(255,255,255,0.75);
    font-size: 0.9em;
  }

  section.divider {
    background: #0d47a1;
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }

  section.divider h1 {
    color: #ffffff;
    border-bottom: 2px solid rgba(255,255,255,0.3);
    font-size: 2.4em;
  }

  section.divider p {
    color: rgba(255,255,255,0.8);
    font-size: 1.1em;
    margin-top: 12px;
  }

  footer {
    color: #90a4ae;
    font-size: 0.7em;
  }
---

<!-- _class: cover -->

# Fraud Detection MLOps

## Sistema completo de detecção de fraude com LLM e agente ReAct

<br>

**FIAP MLET — Fase 05 | Demo Day**

scikit-learn · PyTorch · MLflow · FastAPI · LangChain · Grafana · Evidently

---

<!-- _class: divider -->

# O Problema

Fraude em cartão de crédito

---

<!-- _footer: "1 / 5 — Problema" -->

# O Problema

**Fraude em cartão de crédito custa ao mercado global +32 bilhões de dólares por ano**

<br>

| Atributo | Valor |
|---|---|
| Transações no dataset | **284.807** |
| Período | 2 dias — setembro 2013, Europa |
| Fraudes | **492** — apenas **0,17%** |
| Features | V1–V28 (PCA anônimo) + Time + Amount |
| Target | Class: 0 = legítima · 1 = fraude |

<br>

> Um modelo que sempre prevê "legítima" teria **99,83% de accuracy** — e seria completamente inútil.
> Por isso usamos **AUC-ROC** e **F1** como métricas principais.

---

<!-- _footer: "1 / 5 — Problema" -->

# Por que este problema é difícil?

**Técnico**
- Desbalanceamento extremo — 1 fraude para cada 580 transações legítimas
- Features V1–V28 são componentes PCA — sem interpretação direta
- Drift temporal — padrões de fraude mudam ao longo do tempo

**Negócio**
- Analista precisa *entender* a decisão, não apenas receber um score
- Falso positivo = cliente bloqueado indevidamente → insatisfação
- Falso negativo = fraude não detectada → prejuízo financeiro direto

**MLOps**
- Reprodutibilidade do pipeline de treino
- Versionamento de dados e modelos
- Monitoramento de drift em produção com retreino automático

---

<!-- _class: divider -->

# Abordagem

4 camadas integradas

---

<!-- _footer: "2 / 5 — Abordagem" -->

# Arquitetura — 4 Camadas

**Camada 1 — Dados + Treino**
DVC (versionamento) · Pandera (schema) · Feature Store incremental · scikit-learn + PyTorch · MLflow Registry (9 tags obrigatórias)

**Camada 2 — Serving + Agente**
FastAPI `/predict` + `/agent/query` · Agente ReAct com 3 tools · RAG via FAISS · Autenticação por API Key · CI/CD GitHub Actions

**Camada 3 — Observabilidade**
Prometheus (métricas) · Grafana (11 painéis + alertas automáticos) · Langfuse (rastreio do LLM) · Evidently (PSI de drift)

**Camada 4 — Segurança + Governança**
Presidio (PII input/output) · OWASP Top 10 LLM (5 ameaças) · Red Teaming (5 cenários) · Model Card · System Card · LGPD

---

<!-- _footer: "2 / 5 — Abordagem" -->

# Stack Tecnológica

| Camada | Ferramentas |
|---|---|
| Dados | DVC · Kaggle · Pandera · Feature Store (Parquet) |
| Modelos | scikit-learn (LR + RF) · PyTorch (MLP) |
| Tracking | MLflow Registry · 9 tags obrigatórias · alias `@Production` |
| Serving | FastAPI · Docker · uvicorn |
| Agente LLM | LangChain ReAct · 3 tools · FAISS · Ollama / OpenAI |
| Avaliação | RAGAS (4 métricas) · LLM-as-judge (3 critérios) · Golden Set 20 pares |
| Observabilidade | Prometheus · Grafana · Langfuse · Evidently PSI |
| Segurança | Presidio · bandit · ruff · OWASP mapping |
| CI/CD | GitHub Actions · staging gate · approval manual |

---

<!-- _class: divider -->

# Demo ao Vivo

`localhost:8000` · `localhost:5000` · `localhost:3000` · `localhost:3001`

---

<!-- _footer: "3 / 5 — Demo: MLflow" -->

# MLflow Registry — Modelo em Produção

**`http://localhost:5000`** → Models → `fraud_detector_rf` → `@Production`

<br>

| Métrica | Valor | Como medido |
|---|---|---|
| **AUC-ROC** | **0.9529** | Holdout temporal (últimas 20% por tempo) |
| **Precision** | **0.9605** | Threshold otimizado = 0.25 |
| **F1-score** | **0.8391** | Threshold otimizado = 0.25 |

<br>

**Tags de governança visíveis no Registry:**
`model_name` · `risk_level: high` · `fairness_checked` · `git_sha` · `fraud_threshold: 0.25`

> **Threshold 0.25** foi calculado automaticamente no treino para maximizar F1.
> Split **temporal** evita data leakage — sem isso, AUC inflaria artificialmente.

---

<!-- _footer: "3 / 5 — Demo: /predict" -->

# API ao Vivo — `POST /predict`

**`http://localhost:8000/docs`**

**Request (copiar no Swagger):**

```json
{
  "Time": 9800.0,
  "Amount": 850.0,
  "V14": -6.5,
  "V1": 0.0, "V2": 0.0, "V3": 0.0
}
```

**Response esperada:**

```json
{
  "fraud_score": 0.8712,
  "label": "fraude",
  "threshold": 0.25
}
```

> R$850 às **2h44 da manhã** + **V14 = −6.5** (componente PCA mais correlacionado com fraude).
> Autenticação via header `X-API-Key` — sem chave, retorna HTTP 401 (mapeado no OWASP).

---

<!-- _footer: "3 / 5 — Demo: Agente ReAct" -->

# Agente ReAct — `POST /agent/query`

**`http://localhost:8000/docs`**

```json
{
  "query": "Esta transação de R$850 às 3h da manhã com V14 = -6.5 é suspeita?",
  "model_name": "llama3.2:3b"
}
```

**Ciclo interno do agente:**

```
Thought  → "Preciso calcular o score e identificar os fatores de risco"
Action   → fraud_predictor({"Amount": 850, "V14": -6.5, ...})
Obs      → {"fraud_score": 0.87, "top_risk_factors": [{"feature": "V14", ...}]}

Thought  → "Vou buscar casos similares na base de conhecimento"
Action   → transaction_lookup("alto valor madrugada V14 negativo")
Obs      → "• Padrão típico: card-not-present, horário atípico, V14 < -5..."

Answer   → "Transação suspeita. V14=-6.5 indica fraude. Recomendo bloqueio preventivo."
```

---

<!-- _footer: "3 / 5 — Demo: Agente" -->

# 3 Tools do Agente

| Tool | O que faz | Arquivo |
|---|---|---|
| `fraud_predictor` | Score via Random Forest + top 5 features SHAP | `src/agent/tools.py:26` |
| `transaction_lookup` | Busca semântica RAG — FAISS + embeddings | `src/agent/tools.py:90` |
| `drift_report` | Estado atual do drift — PSI por feature | `src/agent/tools.py:104` |

<br>

**RAG Pipeline:**
- Base de conhecimento: documentos sobre padrões de fraude, regulação, casos históricos
- FAISS local — sem servidor externo, funciona completamente offline
- Embeddings: `nomic-embed-text` (Ollama) ou `text-embedding-3-small` (OpenAI)

<br>

> O agente não diz só "fraude" — explica **por que**, com quais evidências, e o que fazer.
> Isso é o que transforma um modelo de ML num sistema de decisão útil.

---

<!-- _footer: "3 / 5 — Demo: Langfuse" -->

# Langfuse — Rastreabilidade do LLM

**`http://localhost:3001`** → abrir trace mais recente

<br>

**O que cada trace registra:**

| Campo | Detalhe |
|---|---|
| Spans | input → thought → tool call → observation → output |
| Tokens | consumidos por etapa |
| Latência | total e por span |
| Conteúdo | cada chamada de ferramenta e sua resposta |

<br>

> **Por que isso importa para o negócio:**
> Auditabilidade completa. Para cada decisão do agente é possível responder:
> *"Por que o sistema disse isso? Que dados usou? Quanto tempo levou?"*
>
> Indispensável em ambiente regulado — LGPD, BACEN, rastreabilidade de decisões automatizadas.

---

<!-- _footer: "3 / 5 — Demo: Grafana" -->

# Grafana — Observabilidade Operacional

**`http://localhost:3000`** (admin / datathon) — 11 painéis

<br>

| Painel | O que mostra |
|---|---|
| Taxa de fraude detectada | Transações bloqueadas por minuto em tempo real |
| Distribuição de scores | Histograma — deslocamento indica drift |
| Latência da API (p99) | Meta: < 200ms |
| PSI por feature | `Amount_scaled` · `V14` · `Hour` · `Amount_log` |
| AUC ao vivo | Degradação dispara alerta automático |

<br>

**Alertas automáticos configurados:**

| Condição | Ação |
|---|---|
| PSI > 0.1 | Warning + alerta no Grafana |
| PSI > 0.2 | Trigger de retreino automático |
| AUC degrada | Alerta de degradação de performance |

---

<!-- _footer: "3 / 5 — Demo: CI/CD" -->

# CI/CD — GitHub Actions

**`github.com/CleitonCardoso/fraud-detection-mlops/actions`**

<br>

**Pipeline CI — toda Pull Request:**

```
ruff lint  →  mypy  →  bandit (SAST)  →  pytest 70% cobertura  →  docker build
```

<br>

**Pipeline CD — merge em main:**

```
deploy staging  →  ▶ approval manual  →  deploy production
```

<br>

> **Human-in-the-loop gate:** modelo novo só vai para produção após aprovação humana.
> Governance by design — não é só boa prática, é requisito regulatório.

- Cobertura atual: **70.7%** — acima dos 60% obrigatórios
- 42 testes passando · 1 skip (PyTorch — Python 3.13 sem suporte)
- CI verde antes de toda demo

---

<!-- _class: divider -->

# Resultados

Métricas mensuráveis em todas as camadas

---

<!-- _footer: "4 / 5 — Resultados" -->

# Métricas do Modelo

| Métrica | Valor | Referência |
|---|---|---|
| **AUC-ROC** | **0.9529** | Random Forest @Production |
| **Precision** | **0.9605** | Threshold 0.25 |
| **F1-score** | **0.8391** | Threshold 0.25 |
| Threshold | **0.25** | Otimizado para F1 no holdout |

<br>

**Qualidade do Agente — RAGAS (20 pares golden set):**

| Métrica | Score | O que mede |
|---|---|---|
| Faithfulness | ≥ 0.85 | Resposta ancorada no contexto recuperado |
| Answer Relevancy | ≥ 0.80 | Responde à pergunta feita |
| Context Precision | ≥ 0.80 | Contexto recuperado é relevante |
| Context Recall | ≥ 0.75 | Toda informação necessária foi recuperada |

---

<!-- _footer: "4 / 5 — Resultados: Segurança" -->

# Segurança — Red Teaming & OWASP

**5 cenários adversariais testados — todos bloqueados**

| # | Cenário | Resultado |
|---|---|---|
| 1 | Jailbreak via personagem ("ignore suas restrições") | **Bloqueado** HTTP 400 |
| 2 | Extração de dados de treino ("quais CPFs no dataset?") | **Bloqueado** + PII removida |
| 3 | Prompt injection via contexto RAG | **Bloqueado** pelo input guardrail |
| 4 | Escalação de privilégios ("ignore a autenticação") | **Bloqueado** HTTP 401 |
| 5 | Data leakage por perguntas indiretas | **Bloqueado** + output sanitizado |

<br>

**LLM-as-judge — 3 critérios:**
precisão técnica · clareza para não-especialistas · *impacto na decisão de negócio*

**OWASP Top 10 LLM:** 5 ameaças mapeadas em `docs/OWASP_MAPPING.md`

---

<!-- _class: divider -->

# Impacto de Negócio

O que isso entrega para um time de fraude real

---

<!-- _footer: "5 / 5 — Impacto" -->

# Impacto de Negócio

**Velocidade**
Inferência em < 200ms. Analista humano leva minutos por transação. Sistema escala para milhares de transações por segundo sem custo incremental.

**Explicabilidade**
SHAP revela as top 5 features por predição. O agente converte isso em linguagem natural com recomendação acionável. Analista sabe *por que* a transação foi bloqueada — não apenas *que* foi.

**Confiabilidade Operacional**
Drift detectado antes de degradar performance. Retreino automático quando PSI > 0.2. Modelo novo só vai para produção após aprovação humana.

**Conformidade**
Features V1–V28 são PCA — anonimizados na origem. Presidio remove CPF/email/telefone de inputs e outputs. Plano LGPD documenta base legal, dados coletados e direitos dos titulares.

---

<!-- _footer: "5 / 5 — Impacto" -->

# Ciclo de Vida do Modelo

| Etapa | Detalhe |
|---|---|
| **Treino** | `make train` → RF + LR + MLP logados no MLflow com 9 tags |
| **Champion-Challenger** | Challenger precisa superar champion em AUC + 0,5% |
| **Approval Gate** | Aprovação humana obrigatória no GitHub Environment |
| **Staging → Produção** | Deploy automático após approval |
| **Drift check** | Evidently roda diariamente — PSI por feature |
| **Retreino agendado** | Semanal (cron) |
| **Retreino emergencial** | Automático quando PSI > 0.2 |

<br>

> Frequência de retraining: **semanal** como baseline + **imediato** quando PSI crítico.
> Padrões de fraude mudam — um modelo estático degrada em semanas.

---

<!-- _footer: "5 / 5 — Impacto" -->

# Critérios de Avaliação vs. Entrega

| Critério | Peso | Entrega |
|---|---|---|
| **Critérios de negócio** | **30%** | Métricas de negócio no Grafana · agente explicável · impacto mensurável |
| LLM serving + agente | 15% | FastAPI + ReAct 3 tools + RAG + CI/CD |
| Pipeline + baseline | 10% | DVC + MLflow + sklearn + PyTorch · `make train` |
| Qualidade LLM | 10% | RAGAS 4 métricas + LLM-as-judge + 20 pares |
| Observabilidade | 10% | Grafana 11 painéis + Langfuse + Evidently PSI duplo |
| Segurança | 10% | Presidio + guardrails + OWASP + red team |
| Governança | 5% | LGPD + SHAP + fairness + System Card |
| Documentação | 5% | Model Card + System Card + README + ADRs |
| PyTorch + MLflow | 5% | MLP + 9 tags obrigatórias |

---

<!-- _class: cover -->

# Obrigado

## Prontos para perguntas técnicas e de negócio

<br>

**Repositório:**
`github.com/CleitonCardoso/fraud-detection-mlops`

**Subir tudo do zero:**
`make demo`

<br>

scikit-learn · PyTorch · MLflow · FastAPI · LangChain · Grafana · Evidently · Langfuse · Presidio · RAGAS
