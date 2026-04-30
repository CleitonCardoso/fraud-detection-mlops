# Datathon — Fraud Detection MLOps System

> **Fase 05 — LLMs e Agentes | Projeto Integrador (Fases 01–05)**
> **Prazo**: 4 dias | **Domínio**: Fintech — Detecção de Fraude

---

## O que é o Kaggle

Kaggle é uma plataforma de competições de Machine Learning e ciência de dados mantida pelo Google. Para este projeto usamos apenas como **fonte de dados pública e gratuita** — não é uma competição.

```mermaid
flowchart LR
    K([Kaggle\nPlataforma]) --> D[Datasets Públicos]
    K --> C[Competitions]
    K --> N[Notebooks GPU]
    K --> L[Leaderboards]

    D --> |"usamos só isso"| DS["Credit Card Fraud\ncreditcard.csv\n284k transações · 144 MB"]

    style DS fill:#4CAF50,color:#fff
    style D fill:#2196F3,color:#fff
```

**Como acessar**: conta gratuita em kaggle.com → Download direto ou via CLI. Não é necessário participar de nenhuma competição.

---

## Por que este dataset?

```mermaid
mindmap
  root((Credit Card\nFraud Dataset))
    Domínio
      Fintech real
      Problema central em pagamentos digitais
      Fraude é o problema central
    Técnico
      284k transações
      Altamente desbalanceado 0.17%
      Features V1-V28 via PCA
      Time + Amount
    MLOps
      Drift natural ao longo do tempo
      Imbalanced = métricas ricas AUC F1
      PCA simula conformidade LGPD
    Agente LLM
      Explicar predições em linguagem natural
      Consultar casos similares via RAG
      Resumir relatórios de drift
      Calcular risco de novas transações
```

---

## O que vamos construir

Um sistema MLOps completo de detecção de fraude com quatro camadas integradas, cobrindo todas as fases do curso.

```mermaid
flowchart TD
    subgraph E1["Etapa 1 — Dados + Baseline (Fases 01-02)"]
        direction LR
        RAW[creditcard.csv\nKaggle] --> DVC[DVC\nVersionamento]
        DVC --> EDA[EDA\nNotebook]
        EDA --> FE[Feature\nEngineering]
        FE --> FS[Feature Store\nMaterialização incremental]
        FS --> BASE[Baseline\nsklearn]
        FS --> MLP[MLP\nPyTorch]
        BASE --> MLF[MLflow\nTracking + Registry\n9 tags obrigatórias]
        MLP --> MLF
    end

    subgraph E2["Etapa 2 — LLM + Agente (Fases 03-05)"]
        direction LR
        MLF --> API[FastAPI\n/predict /agent]
        API --> AGT[Agente ReAct\nLangChain]
        AGT --> T1[Tool 1\nFraud Predictor]
        AGT --> T2[Tool 2\nTransaction Lookup]
        AGT --> T3[Tool 3\nDrift Report]
        T2 --> RAG[RAG Pipeline\nFAISS + Embeddings]
        API --> CI[CI/CD\nGitHub Actions]
    end

    subgraph E3["Etapa 3 — Avaliação + Observabilidade (Fases 03-05)"]
        direction LR
        GS[Golden Set\n≥20 pares] --> RG[RAGAS\n4 métricas]
        GS --> LJ[LLM-as-judge\n3 critérios]
        API --> LF[Langfuse\nTelemetria]
        API --> PM[Prometheus\nMétricas]
        PM --> GR[Grafana\nDashboard]
        API --> EV[Evidently\nDrift Detection]
    end

    subgraph E4["Etapa 4 — Segurança + Governança (Fases 04-05)"]
        direction LR
        GD[Guardrails\nInput + Output] --> OW[OWASP Top 10\n≥5 ameaças]
        OW --> RT[Red Teaming\n≥5 cenários]
        RT --> LG[LGPD Plan]
        LG --> SC[System Card\n+ Model Card]
    end

    E1 --> E2
    E2 --> E3
    E2 --> E4
    E4 --> DEMO((Demo Day\n≤10 min))

    style DEMO fill:#4CAF50,color:#fff
    style E1 fill:#fff3e0
    style E2 fill:#e3f2fd
    style E3 fill:#f3e5f5
    style E4 fill:#fce4ec
```

---

## Como vamos ingerir os dados

```mermaid
flowchart TD
    START([Início]) --> CHOICE{Método de\ningestion}

    CHOICE --> |"Recomendado\npara CI/CD"| CLI[Kaggle CLI]
    CHOICE --> |"Alternativa\nmanual"| MANUAL[Download no Browser]

    CLI --> INSTALL["pip install kaggle"]
    INSTALL --> CRED["Criar ~/.kaggle/kaggle.json\ncom username + API key"]
    CRED --> DOWN["kaggle datasets download\n-d mlg-ulb/creditcardfraud\n-p data/raw/ --unzip"]

    MANUAL --> SITE["kaggle.com/datasets/\nmlg-ulb/creditcardfraud"]
    SITE --> DLBTN["Clicar Download\n(conta gratuita)"]
    DLBTN --> PLACE["Mover creditcard.csv\npara data/raw/"]

    DOWN --> CSV["data/raw/creditcard.csv\n144 MB · 284.807 linhas"]
    PLACE --> CSV

    CSV --> DVC_ADD["dvc add data/raw/creditcard.csv"]
    DVC_ADD --> GIT_ADD["git add data/raw/creditcard.csv.dvc\n         data/raw/.gitignore"]
    GIT_ADD --> COMMIT["git commit -m 'track: raw fraud dataset'"]
    COMMIT --> PUSH["dvc push  →  storage remoto"]

    PUSH --> REPRO([Qualquer pessoa reproduz\ncom 'dvc pull'])

    style CSV fill:#4CAF50,color:#fff
    style REPRO fill:#2196F3,color:#fff
```

### Por que DVC e não Git?

| O que **não** fazer | O que **fazemos** |
|---|---|
| `git add creditcard.csv` (144 MB no repo) | `dvc add creditcard.csv` (só o hash no Git) |
| Dados diferentes em cada máquina | `dvc pull` reproduz exatamente os mesmos dados |
| Impossível auditar versão dos dados | `.dvc` file registra hash + metadata |
| Falha em CI por falta de dados | Pipeline DVC como etapa do workflow |

---

## Plano de 4 dias

```mermaid
gantt
    title Datathon — 4 dias de implementação
    dateFormat  YYYY-MM-DD
    axisFormat  Dia %d

    section Dia 1 · Fundação
    Estrutura do projeto        :d1a, 2026-04-29, 2h
    Ingestão + DVC              :d1b, after d1a, 1h
    EDA notebook                :d1c, after d1b, 2h
    Feature engineering         :d1d, after d1c, 2h
    Baseline sklearn            :d1e, after d1d, 1h
    MLP PyTorch                 :d1f, after d1e, 1h
    MLflow tracking             :d1g, after d1f, 1h

    section Dia 2 · LLM + Agente
    FastAPI endpoints           :d2a, 2026-04-30, 2h
    Docker + compose            :d2b, after d2a, 1h
    Agente ReAct + 3 tools      :d2c, after d2b, 3h
    RAG pipeline                :d2d, after d2c, 2h
    GitHub Actions CI           :d2e, after d2d, 1h

    section Dia 3 · Eval + Observabilidade
    Golden Set 20 pares         :d3a, 2026-05-01, 2h
    RAGAS 4 métricas            :d3b, after d3a, 1h
    LLM-as-judge                :d3c, after d3b, 1h
    Langfuse telemetria         :d3d, after d3c, 1h
    Prometheus + Grafana        :d3e, after d3d, 2h
    Evidently drift             :d3f, after d3e, 1h

    section Dia 4 · Segurança + Docs
    Guardrails input/output     :d4a, 2026-05-02, 2h
    OWASP mapping               :d4b, after d4a, 1h
    Red teaming 5 cenários      :d4c, after d4b, 1h
    LGPD plan                   :d4d, after d4c, 1h
    Model Card + System Card    :d4e, after d4d, 2h
    Ensaio pitch                :d4f, after d4e, 1h
```

---

### Dia 1 — Fundação (Fases 01–02)

**Objetivo**: dados versionados, EDA documentada, baseline treinado e rastreado no MLflow.

```mermaid
flowchart LR
    A[Download\nKaggle] --> B[DVC\nAdd + Push]
    B --> C[EDA\nNotebook]
    C --> D[Feature\nEngineering\n+ Pandera]
    D --> FS[Feature Store\nMaterialização\nincremental]
    FS --> E[Baseline\nsklearn]
    FS --> F[MLP\nPyTorch]
    E --> G[MLflow\nRegistry\n9 tags]
    F --> G
    G --> H{AUC ≥ 0.90?}
    H --> |Sim| OK1([Dia 1 ✓])
    H --> |Não| I[Tunar\nhiperparâmetros]
    I --> E

    style OK1 fill:#4CAF50,color:#fff
    style FS fill:#9C27B0,color:#fff
    style G fill:#2196F3,color:#fff
```

| Tarefa | Entregável |
|---|---|
| Estrutura do projeto | `pyproject.toml`, `Makefile`, `.env.example`, pastas |
| Ingestão + DVC | `data/raw/creditcard.csv.dvc` commitado |
| EDA (`notebooks/01_eda.ipynb`) | Distribuição de classes, correlações, análise temporal |
| Feature engineering (`src/features/`) | Scaling, features derivadas, schema Pandera validado |
| **Feature Store** (`src/features/feature_store.py`) | Materialização incremental via upsert — **nunca full-flush**; features compartilhadas entre baseline e MLP |
| **Dados sintéticos** (`tests/conftest.py`) | Faker + fixtures com distribuição realista de fraude — usado em todos os testes, nunca dados reais |
| Baseline sklearn (`src/models/baseline.py`) | LogisticRegression + RandomForest |
| MLP PyTorch (`src/models/mlp.py`) | Rede simples, BCE loss, class weights para desbalanceamento |
| **MLflow Registry** (`src/models/train.py`) | 9 tags obrigatórias: `model_name`, `model_version`, `model_type`, `training_data_version`, `metrics`, `owner`, `risk_level`, `fairness_checked`, `git_sha` |
| Testes (`tests/test_features.py`, `tests/test_models.py`) | pytest com dados sintéticos, cobertura ≥ 60% |

**Schema obrigatório de tags no MLflow Registry:**
```python
required_tags = {
    "model_name":              str,   # ex: "fraud_detector_rf"
    "model_version":           str,   # ex: "1.2.0"
    "model_type":              str,   # "classification"
    "training_data_version":   str,   # hash DVC do creditcard.csv
    "metrics":                 dict,  # {"auc": 0.95, "f1": 0.88}
    "owner":                   str,   # email do responsável
    "risk_level":              str,   # "high" (decisão financeira)
    "fairness_checked":        bool,  # True após análise fairlearn
    "git_sha":                 str,   # commit hash
}
```

**Critério de saída**: `make train` roda do zero, todas as 9 tags aparecem no MLflow Registry.

---

### Dia 2 — LLM + Agente (Fases 03 + 05)

**Objetivo**: agente ReAct funcional, servido via FastAPI, com CI/CD rodando.

```mermaid
flowchart TD
    subgraph API["FastAPI"]
        EP1["/predict\nModelo sklearn/PyTorch"]
        EP2["/agent/query\nAgente ReAct"]
        EP3["/health\nLiveness check"]
        EP4["/metrics\nPrometheus scrape"]
    end

    subgraph AGENT["Agente ReAct"]
        Q[Query do usuário] --> THINK[Thought]
        THINK --> ACT{Escolhe\nTool}
        ACT --> T1["fraud_predictor\nScore + SHAP"]
        ACT --> T2["transaction_lookup\nRAG: casos similares"]
        ACT --> T3["drift_report\nEstado atual do drift"]
        T1 --> OBS[Observation]
        T2 --> OBS
        T3 --> OBS
        OBS --> THINK
        THINK --> |"Final Answer"| RESP[Resposta ao usuário]
    end

    EP2 --> AGENT

    subgraph RAG["RAG Pipeline"]
        KB[Base de conhecimento\nFraude + Regulação] --> EMB[Embeddings\nOpenAI]
        EMB --> FAISS[FAISS\nVector Store]
        FAISS --> RET[Retriever\nTop-K chunks]
    end

    T2 --> RAG

    subgraph CI["GitHub Actions — CI + CD com Staging"]
        lint[ruff lint] --> types[mypy]
        types --> sec[bandit]
        sec --> test[pytest ≥60%]
        test --> build[docker build + push]
        build --> staging[deploy → staging\nCloud Run staging]
        staging --> gate{Approval\nManual}
        gate --> |Aprovado| prod[deploy → production\nCloud Run prod]
        gate --> |Rejeitado| halt[Pipeline\ninterrompido]
    end

    style API fill:#e3f2fd
    style AGENT fill:#fff3e0
    style RAG fill:#f3e5f5
    style CI fill:#e8f5e9
    style gate fill:#FF9800,color:#fff
    style prod fill:#4CAF50,color:#fff
    style halt fill:#f44336,color:#fff
```

| Tarefa | Entregável |
|---|---|
| FastAPI (`src/serving/app.py`) | Endpoints `/predict`, `/agent/query`, `/health`, `/metrics` |
| Dockerfile + docker-compose | serving + MLflow + Prometheus + Grafana + Langfuse |
| Agente ReAct (`src/agent/react_agent.py`) | LangChain/LangGraph, max_iterations=10 |
| Tool 1: `fraud_predictor` | Retorna score + explicação SHAP em linguagem natural |
| Tool 2: `transaction_lookup` | RAG sobre base de casos de fraude documentados |
| Tool 3: `drift_report` | Lê relatório Evidently e resume o estado do drift |
| RAG pipeline (`src/agent/rag_pipeline.py`) | FAISS local, embeddings OpenAI |
| **CI/CD com staging** (`.github/workflows/`) | `ci.yml`: lint → mypy → bandit → pytest → docker build · `cd.yml`: deploy staging → **approval manual** → deploy production |
| **Ambiente staging** | Cloud Run service separado (`fraud-api-staging`) — recebe todo merge em `main` antes da produção |
| **Quantização do LLM** | Benchmark: gpt-4o-mini (API) vs. llama3.2:3b-q4\_K\_M (Ollama local) — latência e qualidade documentadas |
| **Benchmark ≥ 3 configurações** (`evaluation/benchmark_configs.py`) | Config A: gpt-4o-mini + temp 0.0 · Config B: gpt-4o-mini + temp 0.3 + chunk size 512 · Config C: Ollama quantizado |

**Critério de saída**: merge em `main` dispara staging automaticamente; produção só recebe após approval manual no GitHub.

---

### Dia 3 — Avaliação + Observabilidade (Fases 03–05)

**Objetivo**: métricas de qualidade calculadas, dashboard ao vivo, drift detectado.

```mermaid
flowchart LR
    subgraph EVAL["Avaliação de Qualidade"]
        GS["Golden Set\n≥20 pares\n(query, expected, contexts)"]
        GS --> RG["RAGAS\n· faithfulness\n· answer_relevancy\n· context_precision\n· context_recall"]
        GS --> LJ["LLM-as-judge\n· precisão técnica\n· clareza\n· acionabilidade"]
    end

    subgraph OBS["Observabilidade Operacional"]
        LF["Langfuse\nTracing por chamada\nTokens · Latência · Spans"]
        PM["Prometheus\nprediction_latency\nfraud_score_dist\nrequest_count"]
        PM --> GR["Grafana Dashboard\nMétricas tempo real"]
    end

    subgraph DRIFT["Drift Detection — PSI duplo threshold"]
        EV["Evidently\nroda diariamente\nPSI por feature + prediction drift"]
        EV --> TH1{"PSI > 0.1?"}
        TH1 --> |Sim| WARN[Warning:\nLogar + notificar\nGrafana alert]
        TH1 --> |Não| OK[Status: estável]
        WARN --> TH2{"PSI > 0.2?"}
        TH2 --> |Sim| ALT[Retrain trigger\nautomático]
        TH2 --> |Não| MON2[Monitorar\ncontinuamente]
    end

    style EVAL fill:#f3e5f5
    style OBS fill:#e3f2fd
    style DRIFT fill:#fff8e1
    style ALT fill:#f44336,color:#fff
    style WARN fill:#FF9800,color:#fff
    style OK fill:#4CAF50,color:#fff
```

| Tarefa | Entregável |
|---|---|
| Golden Set (`data/golden_set/`) | ≥ 20 pares query/resposta sobre fraude e o sistema |
| RAGAS (`evaluation/ragas_eval.py`) | 4 métricas: faithfulness, answer_relevancy, context_precision, context_recall |
| LLM-as-judge (`evaluation/llm_judge.py`) | 3 critérios: **precisão técnica** · **clareza para não-especialistas** · **impacto na decisão de negócio** (critério de negócio obrigatório pela rubrica) |
| Langfuse | Tracing completo de cada chamada do agente (tokens, latência, spans) |
| Prometheus (`src/monitoring/metrics.py`) | 3 métricas customizadas expostas em `/metrics` |
| Grafana | Dashboard com ≥ 4 painéis — Langfuse + Prometheus integrados, funcionando **simultaneamente durante a demo**; **alertas automáticos** configurados por degradação de AUC e PSI |
| **Evidently com PSI duplo** (`src/monitoring/drift.py`) | PSI > 0.1 → warning + Grafana alert · PSI > 0.2 → retrain trigger automático · métricas de drift logadas no MLflow |

**Critério de saída**: Grafana E Langfuse mostram dados ao vivo ao mesmo tempo; Evidently detecta os dois níveis de PSI e dispara alertas corretos.

---

### Dia 4 — Segurança + Governança (Fases 04–05)

**Objetivo**: sistema defensável perante banca técnica e de negócio.

```mermaid
flowchart TD
    INPUT[Query do usuário] --> IG{Input\nGuardrail}

    IG --> |"Prompt injection\ndetectado"| BLOCK1[Bloquear\nLogar + alertar]
    IG --> |"PII no input\ndetectado"| ANON1[Anonimizar\nPresidio]
    IG --> |"Tamanho > 4096\nchars"| BLOCK2[Bloquear\nErro 400]
    IG --> |"Input válido"| LLM[Agente LLM\nProcessa]

    LLM --> OG{Output\nGuardrail}
    OG --> |"PII no output"| ANON2[Anonimizar\nPresidio]
    OG --> |"Output limpo"| RESP[Resposta\nao usuário]

    subgraph RED["Red Teaming — 5 cenários testados"]
        RT1[Jailbreak via\npersonagem]
        RT2[Extração de\ndados de treino]
        RT3[Prompt injection\nvia contexto RAG]
        RT4[Escalação de\nprivilégios]
        RT5[Data leakage via\nperguntas indiretas]
    end

    subgraph DOCS["Documentação de Governança"]
        MC[Model Card\nMétricas + Limitações]
        SC[System Card\nRiscos Sistêmicos]
        LG[LGPD Plan\nMapeamento de dados]
        OW[OWASP Mapping\n≥5 ameaças]
    end

    style BLOCK1 fill:#f44336,color:#fff
    style BLOCK2 fill:#f44336,color:#fff
    style ANON1 fill:#FF9800,color:#fff
    style ANON2 fill:#FF9800,color:#fff
    style RESP fill:#4CAF50,color:#fff
```

| Tarefa | Entregável |
|---|---|
| Input guardrail (`src/security/guardrails.py`) | Detecta injection, limita tamanho, anonimiza PII |
| Output guardrail | Presidio remove CPF/email/telefone do output |
| OWASP mapping (`docs/OWASP_MAPPING.md`) | ≥ 5 ameaças do OWASP Top 10 LLM com mitigação |
| Red teaming (`docs/RED_TEAM_REPORT.md`) | ≥ 5 cenários adversariais documentados e testados |
| LGPD plan (`docs/LGPD_PLAN.md`) | Base legal, dados coletados, direitos dos titulares |
| Model Card (`docs/MODEL_CARD.md`) | Métricas, limitações, viés, uso pretendido |
| System Card (`docs/SYSTEM_CARD.md`) | Arquitetura, riscos sistêmicos, responsáveis |
| Fairness check | Viés por segmento (valor, horário) via `fairlearn` |
| Ensaio do pitch | ≤ 10 min com timer, backup de slides |

**Critério de saída**: checklist completo, `make demo` sobe tudo do zero.

---

## Estrutura de pastas

```
datathon/
├── .github/
│   └── workflows/
│       ├── ci.yml                   # lint → type-check → test → docker build
│       └── cd.yml                   # deploy staging → approval gate → deploy production
├── data/
│   ├── raw/                         # creditcard.csv (DVC, não no Git)
│   ├── processed/                   # features processadas (DVC)
│   └── golden_set/                  # ≥20 pares para avaliação RAG
├── src/
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── feature_store.py          # materialização incremental (upsert, nunca full-flush)
│   ├── models/
│   │   ├── baseline.py              # sklearn (LogReg, RandomForest)
│   │   ├── mlp.py                   # PyTorch MLP
│   │   └── train.py                 # pipeline MLflow
│   ├── agent/
│   │   ├── react_agent.py           # agente ReAct (LangChain)
│   │   ├── tools.py                 # ≥3 tools customizadas
│   │   └── rag_pipeline.py          # FAISS + embeddings
│   ├── serving/
│   │   ├── app.py                   # FastAPI
│   │   └── Dockerfile
│   ├── monitoring/
│   │   ├── drift.py                 # Evidently
│   │   └── metrics.py               # Prometheus
│   └── security/
│       ├── guardrails.py            # input + output
│       └── pii_detection.py         # Presidio
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_agent.py
│   ├── test_api.py
│   └── test_guardrails.py
├── evaluation/
│   ├── ragas_eval.py
│   ├── llm_judge.py
│   └── golden_set_builder.py
├── notebooks/
│   └── 01_eda.ipynb
├── docs/
│   ├── MODEL_CARD.md
│   ├── SYSTEM_CARD.md
│   ├── LGPD_PLAN.md
│   ├── OWASP_MAPPING.md
│   └── RED_TEAM_REPORT.md
├── configs/
│   ├── model_config.yaml
│   └── monitoring_config.yaml
├── docker-compose.yml
├── dvc.yaml
├── pyproject.toml
├── Makefile
├── .env.example
└── README.md
```

---

## Stack tecnológica

```mermaid
graph TB
    subgraph DATA["Camada de Dados"]
        KG[Kaggle\nCreditCard CSV]
        DVC_S[DVC\nVersionamento]
        PAN[Pandera\nSchema validation]
    end

    subgraph TRAIN["Camada de Treino"]
        SKL[scikit-learn\nLogReg + RandomForest]
        PT[PyTorch\nMLP]
        MLF_S[MLflow\nTracking + Registry]
    end

    subgraph SERVE["Camada de Serving"]
        FA[FastAPI\nREST API]
        DOC[Docker\nContainer]
        LC[LangChain\nReAct Agent]
        FAISS_S[FAISS\nVector Store]
    end

    subgraph EVAL_S["Camada de Avaliação"]
        RG_S[RAGAS\n4 métricas]
        LJJ[LLM-as-judge\n3 critérios]
        LF_S[Langfuse\nTelemetria LLM]
    end

    subgraph MON["Camada de Monitoramento"]
        PR[Prometheus\nMétricas]
        GR_S[Grafana\nDashboard]
        EV_S[Evidently\nDrift PSI]
    end

    subgraph SEC["Camada de Segurança"]
        GRD[Guardrails\nInput/Output]
        PRES[Presidio\nPII Detection]
        BAN[Bandit\nSAST]
    end

    subgraph CICD["CI/CD"]
        GHA[GitHub Actions\nlint → test → build]
    end

    KG --> DVC_S --> PAN --> SKL & PT
    SKL & PT --> MLF_S --> FA
    FA --> DOC --> LC --> FAISS_S
    FA --> PR --> GR_S
    FA --> LF_S --> RG_S & LJJ
    FA --> EV_S
    FA --> GRD --> PRES
    GHA --> DOC

    style DATA fill:#fff3e0
    style TRAIN fill:#e8f5e9
    style SERVE fill:#e3f2fd
    style EVAL_S fill:#f3e5f5
    style MON fill:#fce4ec
    style SEC fill:#fff8e1
    style CICD fill:#e0f2f1
```

| Camada | Ferramenta | Justificativa |
|---|---|---|
| Dados | DVC + Kaggle CLI | Reprodutível, sem dado no Git |
| Validação | Pandera | Schema contracts em feature engineering |
| **Feature Store** | Parquet + upsert incremental | MLOps Maturity Level 2 — sem full-flush |
| **Dados sintéticos** | Faker + fixtures pytest | Dev/test sem dados reais; GAP 08 da rubrica |
| Baseline | scikit-learn + PyTorch | Rubrica exige ambos explicitamente |
| Tracking | MLflow + Registry (9 tags) | Padrão da rubrica com schema obrigatório |
| Serving | FastAPI + Docker | Leve, tipado, fácil de testar |
| **CI/CD com staging** | GitHub Actions + Environments | lint → test → staging → **approval** → production |
| Agente | LangChain + LangGraph | Suporte nativo a ReAct e tools |
| LLM | OpenAI API (gpt-4o-mini) | Custo baixo, sem GPU local |
| RAG | FAISS + OpenAI Embeddings | Sem servidor externo, funciona local |
| Avaliação | RAGAS + LLM-as-judge | Rubrica exige ambos |
| Telemetria | Langfuse | LLMOps nativo, open-source |
| Monitoramento | Prometheus + Grafana + **alertas** | Stack padrão da Fase 03 com alertas automáticos |
| Drift | Evidently + PSI duplo (0.1/0.2) | Padrão da Fase 04 com dois níveis de threshold |
| Segurança | Presidio + regex | PII detection + injection |

---

## Critérios de avaliação e como atendemos cada um

```mermaid
pie title Distribuição de Pesos — Datathon Fase 05
    "Critérios de negócio (empresa)" : 30
    "LLM serving + agente" : 15
    "Pipeline + baseline" : 10
    "Avaliação de qualidade LLM" : 10
    "Observabilidade + monitoramento" : 10
    "Segurança + guardrails" : 10
    "Governança + compliance" : 5
    "Documentação + arquitetura" : 5
    "PyTorch + MLflow" : 5
```

| Critério | Peso | Como atendemos |
|---|---|---|
| Critérios de negócio | 30% | Domínio fintech real, métricas de negócio no Grafana (taxa de fraude, valor em risco), demo orientado ao impacto |
| LLM serving + agente | 15% | FastAPI + ReAct com 3 tools + RAG funcional + CI/CD |
| Pipeline + baseline | 10% | DVC + MLflow + sklearn + PyTorch, `make train` reprodutível |
| Qualidade LLM | 10% | RAGAS 4 métricas + LLM-as-judge 3 critérios + golden set 20 pares |
| Observabilidade | 10% | Grafana + Langfuse + Evidently drift + alertas PSI |
| Segurança | 10% | Presidio + guardrails + OWASP ≥5 + red teaming ≥5 |
| Governança | 5% | LGPD + fairness + explicabilidade SHAP + System Card |
| Documentação | 5% | Model Card + System Card + README completo + ADRs |
| PyTorch + MLflow | 5% | MLP treinado + MLflow com tags obrigatórias |

---

## Deployment — Onde e como o sistema fica disponível online

### Onde cada peça roda

```mermaid
flowchart TD
    subgraph DEV["Ambiente local (desenvolvimento)"]
        DC[docker-compose\nFastAPI + MLflow\n+ Prometheus + Grafana\n+ Langfuse]
    end

    subgraph CI["GitHub Actions (CI/CD)"]
        LINT[lint + type-check\n+ security scan\n+ pytest]
        BUILD[docker build\n+ push para\nGitHub Container Registry]
        LINT --> BUILD
    end

    subgraph CLOUD["Cloud — Google Cloud Run (gratuito)"]
        CR[Cloud Run\nFastAPI container\nURL pública HTTPS\nauto-scale · scale-to-zero]
    end

    subgraph MON_CLOUD["Monitoramento online"]
        GC[Grafana Cloud\nFree tier\n10k métricas · 50 GB logs]
        LFC[Langfuse Cloud\nFree tier\n50k traces/mês]
    end

    DEV --> |"git push → PR"| CI
    CI --> |"merge em main"| CLOUD
    CLOUD --> |"métricas via remote_write"| MON_CLOUD

    style CLOUD fill:#4CAF50,color:#fff
    style CI fill:#2196F3,color:#fff
    style MON_CLOUD fill:#9C27B0,color:#fff
```

### Por que Google Cloud Run?

| Critério | Cloud Run | Alternativas descartadas |
|---|---|---|
| **Custo** | Gratuito: 2M requests/mês, 360k GB-s de compute | AWS EC2 (sempre ligado = caro), Heroku (sem free tier) |
| **Deploy** | `gcloud run deploy` com uma linha, ou automático via GitHub Actions | SageMaker (complexo demais para o prazo) |
| **Docker-native** | Recebe qualquer imagem Docker diretamente | Render (mais lento no cold start) |
| **HTTPS automático** | URL pública com SSL já configurado | VPS manual (precisa configurar nginx + certbot) |
| **Scale-to-zero** | Não cobra quando não há requisições | VM sempre ligada cobra 24/7 |

### Fluxo completo de deploy

```mermaid
sequenceDiagram
    participant Dev as Desenvolvedor
    participant GH as GitHub
    participant GA as GitHub Actions
    participant GCR as GitHub Container Registry
    participant CR as Cloud Run
    participant USER as Usuário / Banca

    Dev->>GH: git push (branch)
    GH->>GA: trigger CI workflow
    GA->>GA: ruff · mypy · bandit · pytest
    GA->>GCR: docker build + push (ghcr.io/...)

    Dev->>GH: merge PR → main
    GH->>GA: trigger CD workflow
    GA->>CR: gcloud run deploy\n--image ghcr.io/...\n--region us-central1
    CR-->>GA: URL pública gerada
    GA-->>Dev: deploy concluído ✓

    USER->>CR: POST /agent/query\n{"query": "..."}
    CR-->>USER: {"answer": "..."}
```

### URLs que a banca vai acessar

| Endpoint | URL (exemplo) | O que faz |
|---|---|---|
| `/predict` | `https://fraud-api-xxx.run.app/predict` | Score de fraude para uma transação |
| `/agent/query` | `https://fraud-api-xxx.run.app/agent/query` | Agente ReAct responde em linguagem natural |
| `/health` | `https://fraud-api-xxx.run.app/health` | Liveness check |
| `/docs` | `https://fraud-api-xxx.run.app/docs` | Swagger UI automático do FastAPI |
| Grafana | `https://xxx.grafana.net` | Dashboard de monitoramento |
| Langfuse | `https://cloud.langfuse.com` | Traces do agente |

---

## Atualização do modelo — Com que frequência retreinamos?

### Estratégia de retraining

Usamos três camadas de trigger, do mais simples ao mais sofisticado:

```mermaid
flowchart TD
    subgraph T1["Trigger 1 — Agendado (baseline)"]
        CRON["Cron semanal\nGitHub Actions Schedule\ntoda segunda às 02h00"]
        CRON --> RETRAIN1[Pipeline de\nretraining]
    end

    subgraph T2["Trigger 2 — Drift detectado (event-driven)"]
        EV2[Evidently\nroda a cada 24h]
        EV2 --> PSI{PSI > 0.2\nem alguma\nfeature?}
        PSI --> |Sim| ALERT[Alerta +\nTrigger imediato]
        PSI --> |Não| OK2[Sem ação\nestado estável]
        ALERT --> RETRAIN2[Pipeline de\nretraining]
    end

    subgraph T3["Trigger 3 — Champion-Challenger + Approval Gate (governança)"]
        RETRAIN1 --> CC[Treina\nChallenger]
        RETRAIN2 --> CC
        CC --> COMPARE{AUC challenger\n≥ AUC champion\n+ 0.5%?}
        COMPARE --> |Não| KEEP[Mantém Champion\nRegistra resultado]
        COMPARE --> |Sim| HITL{Human-in-the-loop\nApproval Gate\nGitHub Environment}
        HITL --> |Aprovado| PROMOTE[Promove Challenger\npara Staging → Production\nno MLflow Registry]
        HITL --> |Rejeitado| BLOCK3[Bloqueia deploy\nRegistra motivo]
        PROMOTE --> DEPLOY2[Novo container\ndeployado no\nCloud Run]
    end

    style PROMOTE fill:#4CAF50,color:#fff
    style KEEP fill:#FF9800,color:#fff
    style ALERT fill:#f44336,color:#fff
    style HITL fill:#9C27B0,color:#fff
    style BLOCK3 fill:#f44336,color:#fff
```

### Cadência de operações

| Operação | Frequência | Trigger | Responsável |
|---|---|---|---|
| **Inferência** | Tempo real | Requisição ao `/predict` | Cloud Run |
| **Drift check** | Diário (02h00) | Cron GitHub Actions | Evidently |
| **Warning PSI** | Quando PSI > 0.1 | Drift check diário | Grafana alert |
| **Retraining agendado** | Semanal (seg 02h00) | Cron GitHub Actions | Pipeline DVC |
| **Retraining emergencial** | Quando PSI > 0.2 | Drift detectado | Pipeline DVC |
| **Champion-challenger** | A cada retraining | Pós-treino automático | MLflow Registry |
| **Human-in-the-loop gate** | Quando challenger vence | Pós-challenger evaluation | **Aprovação manual** no GitHub |
| **Deploy de novo modelo** | Após aprovação humana | GitHub Environment approval | GitHub Actions |

### Por que semanal como baseline?

O dataset de fraude tem padrões que mudam com o tempo (novos vetores de ataque, sazonalidade). Semanal é:
- Rápido o suficiente para capturar mudanças relevantes
- Lento o suficiente para não gastar compute desnecessário
- Comum em sistemas de detecção de fraude reais

Para o **Demo Day**, vamos simular o retraining manualmente com `make train` para demonstrar o processo ao vivo.

### Ciclo de vida de uma versão do modelo

```mermaid
stateDiagram-v2
    [*] --> Training : pipeline de treino iniciado
    Training --> Challenger : modelo treinado + 9 tags logadas no MLflow
    Challenger --> Evaluation : métricas calculadas em holdout set
    Evaluation --> Archived : AUC < champion + 0.5%
    Evaluation --> AwaitingApproval : AUC ≥ champion + 0.5%
    AwaitingApproval --> Staging : aprovação humana concedida\n(GitHub Environment gate)
    AwaitingApproval --> Archived : aprovação negada
    Staging --> Production : smoke tests passam no staging
    Production --> Monitoring : deploy no Cloud Run prod
    Monitoring --> Warning : PSI > 0.1
    Warning --> Training : PSI > 0.2 (retrain trigger)
    Warning --> Monitoring : PSI estabiliza
    Archived --> [*]
```

---

## Checklist de entrega

### Etapa 1 — Dados + Baseline
- [ ] EDA documentada com insights sobre o padrão de fraude
- [ ] Baseline treinado, métricas reportadas no MLflow (AUC, F1, precision, recall)
- [ ] Pipeline versionado (DVC + Docker) e reprodutível com `make train`
- [ ] **MLflow Registry com 9 tags obrigatórias** (model_name, model_version, model_type, training_data_version, metrics, owner, risk_level, fairness_checked, git_sha)
- [ ] **Feature Store com materialização incremental** (upsert — nunca full-flush)
- [ ] **Dados sintéticos com Faker** em `tests/conftest.py` — todos os testes usam fixtures, nunca dados reais
- [ ] Métricas de negócio mapeadas (ex: valor total em risco, taxa de detecção)
- [ ] `pyproject.toml` com todas as dependências e constraints

### Etapa 2 — LLM + Agente
- [ ] LLM servido via FastAPI com endpoint documentado
- [ ] **Quantização aplicada e documentada** (gpt-4o-mini via API + Ollama GGUF local comparados)
- [ ] Agente ReAct funcional com ≥ 3 tools relevantes ao domínio
- [ ] RAG retornando contexto de casos de fraude similares
- [ ] **CI/CD com staging**: merge → staging automático → **approval manual** → production
- [ ] **Benchmark documentado com exatamente 3 configurações**: Config A (temp 0.0), Config B (temp 0.3 + chunk 512), Config C (Ollama quantizado)

### Etapa 3 — Avaliação + Observabilidade
- [ ] Golden set com ≥ 20 pares relevantes ao domínio de fraude
- [ ] RAGAS: 4 métricas calculadas e reportadas
- [ ] **LLM-as-judge com ≥ 3 critérios incluindo critério de negócio** (impacto na decisão do analista)
- [ ] **Langfuse + Grafana funcionando simultaneamente** (end-to-end, não só existindo)
- [ ] **Drift com PSI duplo**: PSI > 0.1 = warning + Grafana alert · PSI > 0.2 = retrain trigger automático
- [ ] **Alertas automáticos** configurados no Grafana por degradação de AUC e PSI

### Etapa 4 — Segurança + Governança
- [ ] OWASP mapping com ≥ 5 ameaças e mitigações
- [ ] Guardrails de input e output funcionais e testados
- [ ] ≥ 5 cenários adversariais testados e documentados
- [ ] Plano LGPD aplicado ao dataset e ao sistema
- [ ] Explicabilidade SHAP e fairness documentados
- [ ] System Card e Model Card completos
- [ ] **Human-in-the-loop approval gate** configurado no GitHub Environment antes de todo deploy em produção

### Demo Day
- [ ] Pitch ≤ 10 min: Problema → Abordagem → Demo → Resultados → Impacto
- [ ] Ensaio com timer realizado
- [ ] Backup: slides offline caso a demo falhe
- [ ] Preparado para Q&A técnico (RAGAS scores, PSI, AUC) e de negócio (impacto financeiro)

---

## Comandos principais

```bash
make setup        # cria venv e instala todas as dependências
make data         # dvc pull (ou instrução de download manual)
make train        # treina baseline + MLP, loga no MLflow
make serve        # docker-compose up (FastAPI + MLflow + Prometheus + Grafana + Langfuse)
make test         # pytest com cobertura ≥ 60%
make eval         # roda RAGAS + LLM-as-judge no golden set
make drift        # gera relatório Evidently (train vs. predições recentes)
make demo         # setup completo para Demo Day do zero
```
