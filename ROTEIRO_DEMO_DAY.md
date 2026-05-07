# Roteiro Demo Day — Fraud Detection MLOps
## FIAP MLET Fase 05 | ≤ 10 minutos

---

## Preparação Antes de Entrar na Sala (30 min antes)

### Terminais / abas abertas

```bash
# Terminal 1 — serviços
docker compose up -d
mlflow server --host 0.0.0.0 --port 5000 &
ollama serve &

# Terminal 2 — API
MLFLOW_TRACKING_URI=http://localhost:5000 PYTHONPATH=. \
  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Terminal 3 — drift (opcional, para mostrar ao vivo)
PYTHONPATH=. python -m src.monitoring.drift
```

### Abas do navegador — abrir nesta ordem

| Aba | URL | O que mostrar |
|-----|-----|---------------|
| 1 | http://localhost:8000/docs | Swagger UI da API |
| 2 | http://localhost:5000 | MLflow Registry |
| 3 | http://localhost:3000 | Grafana (admin / datathon) |
| 4 | http://localhost:3001 | Langfuse traces |
| 5 | https://github.com/CleitonCardoso/fraud-detection-mlops | GitHub (CI/CD verde) |

### Editor de texto — deixar aberto com o JSON de teste

```json
{
  "Time": 9800.0,
  "Amount": 850.0,
  "V14": -6.5,
  "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
  "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
  "V11": 0.0, "V12": 0.0, "V13": 0.0, "V15": 0.0, "V16": 0.0,
  "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0, "V21": 0.0,
  "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0, "V26": 0.0,
  "V27": 0.0, "V28": 0.0
}
```

---

## SEGMENTO 1 — Problema (1 min 00 s)

**[Abrir: nenhuma tela específica — falar de frente para a banca]**

> "Fraude em cartão de crédito custa ao mercado global mais de 32 bilhões de dólares por ano.
> O desafio técnico é detectar padrões em milissegundos, num dataset extremamente desbalanceado:
> 284 mil transações, apenas 0,17% são fraude — 492 casos em dois dias de transações europeias.
>
> O problema que esse sistema resolve é duplo:
> Primeiro, o técnico: treinar, versionar, monitorar e retreinar um modelo de ML em produção.
> Segundo, o de negócio: fazer isso de forma que um analista possa confiar na decisão, entender por que uma transação foi bloqueada, e agir com velocidade."

**[Abrir: `notebooks/01_eda.ipynb` — rolar até o gráfico de distribuição de classes]**

> "Aqui está o dataset. Essa assimetria extrema é justamente por que AUC e F1 importam mais que accuracy.
> Um modelo que sempre prevê 'legítima' teria 99,83% de acerto — e seria completamente inútil."

---

## SEGMENTO 2 — Abordagem / Arquitetura (1 min 30 s)

**[Abrir: `docs/ARCHITECTURE.md` — ou mostrar o diagrama do README]**

> "Construímos um sistema com quatro camadas integradas."

Apontar para cada camada enquanto fala:

> "**Camada 1 — Dados e treino**: dataset versionado com DVC, feature engineering com Pandera para contratos de schema, dois modelos — Random Forest com scikit-learn e MLP com PyTorch — rastreados no MLflow com 9 tags obrigatórias de governança.
>
> **Camada 2 — Serving e agente**: FastAPI com dois endpoints principais — `/predict` para inferência síncrona, e `/agent/query` para um agente ReAct com três ferramentas: predição com SHAP, busca semântica via RAG com FAISS, e relatório de drift.
>
> **Camada 3 — Observabilidade**: Prometheus coletando métricas da API, Grafana com 11 painéis e alertas automáticos, Langfuse rastreando cada chamada do LLM, Evidently calculando PSI de drift diariamente.
>
> **Camada 4 — Segurança e governança**: guardrails de input/output com Presidio para PII, mapeamento OWASP com 5 ameaças, 5 cenários de red teaming testados, Model Card, System Card, e plano LGPD."

---

## SEGMENTO 3 — Demo ao Vivo (5 min 30 s)

### 3.1 — Modelo no MLflow Registry (45 s)

**[Ir para aba 2: http://localhost:5000]**

> "Aqui está o MLflow. O modelo em produção é o `fraud_detector_rf` com alias `@Production`."

Clicar em **Models → fraud_detector_rf → versão mais recente**.

> "Vejam as tags: `model_name`, `risk_level: high`, `fairness_checked`, `git_sha`, `fraud_threshold` — esse threshold foi calculado automaticamente no treino para maximizar F1. Hoje está em 0.25, calibrado para o desbalanceamento do dataset."

Mostrar a aba **Metrics**:

> "AUC de 0.95, precision de 0.96, F1 de 0.84 no holdout temporal — split por tempo para evitar data leakage."

### 3.2 — API ao Vivo — Predição (1 min 00 s)

**[Ir para aba 1: http://localhost:8000/docs]**

> "A API está ao vivo. Vou fazer uma predição ao vivo agora."

Clicar em **POST /predict → Try it out**.
Colar o JSON de teste (copiar do editor de texto):

```json
{"Time": 9800.0, "Amount": 850.0, "V14": -6.5, "V1": 0.0, ...}
```

Clicar **Execute**. Mostrar a resposta:

> "Score de fraude: ~0.87. Label: fraude. Threshold: 0.25. Essa transação — R$850 às 2h44 da manhã, com V14 muito negativo, que é o componente PCA mais correlacionado com fraude — foi corretamente classificada.
>
> A autenticação é via header `X-API-Key`. Sem a chave, a API retorna 401. Isso está mapeado no nosso documento OWASP."

### 3.3 — Agente ReAct (1 min 30 s)

**[Ainda na aba 1: http://localhost:8000/docs]**

Clicar em **POST /agent/query → Try it out**. Usar:

```json
{
  "query": "Esta transação de R$850 às 3h da manhã com V14 = -6.5 é suspeita? Quais são os fatores de risco?",
  "model_name": "llama3.2:3b"
}
```

Enquanto aguarda a resposta (5-10 segundos):

> "O agente ReAct está trabalhando agora. Ele segue o ciclo Thought → Action → Observation. Primeiro vai chamar o `fraud_predictor` para calcular o score e os valores SHAP. Depois pode chamar o `transaction_lookup` para buscar casos similares na base de conhecimento via RAG."

Mostrar a resposta do agente:

> "Vejam — o agente explicou a decisão em linguagem natural, citou os fatores de risco específicos com SHAP, e deu uma recomendação acionável. Isso é o que diferencia um modelo de ML de um sistema de decisão útil para o analista."

### 3.4 — Langfuse: Rastreabilidade do LLM (45 s)

**[Ir para aba 4: http://localhost:3001]**

> "Cada chamada do agente gera um trace completo aqui no Langfuse."

Abrir o trace mais recente:

> "Vejam os spans: input do usuário, pensamento do agente, chamada de ferramenta, observação, resposta final. Latência total, tokens consumidos, e o conteúdo de cada passo. Isso é fundamental para LLMOps — auditabilidade de cada decisão do agente."

### 3.5 — Grafana: Observabilidade (1 min 00 s)

**[Ir para aba 3: http://localhost:3000]**

Navegar para o dashboard principal:

> "Grafana com 11 painéis. Deixa eu mostrar os mais relevantes para o negócio."

Apontar para os painéis:

> "**Taxa de fraude detectada** — em tempo real, quantas transações por minuto estão sendo bloqueadas.
>
> **Distribuição de scores** — histograma dos scores de fraude. Se isso começar a se deslocar, é sinal de drift.
>
> **Latência da API** — p99 abaixo de 200ms. FastAPI + scikit-learn é extremamente eficiente.
>
> **PSI por feature** — aqui está o monitoramento de drift. Threshold warning em 0.1, threshold de retreino em 0.2. Quando PSI ultrapassa 0.2, um GitHub Actions dispara o pipeline de retreino automaticamente."

Mostrar os alertas configurados:

> "Quatro alertas configurados: degradação de AUC, PSI crítico, latência alta, e taxa de erro. Isso é um sistema operacional, não só um modelo."

### 3.6 — CI/CD no GitHub (30 s)

**[Ir para aba 5: https://github.com/CleitonCardoso/fraud-detection-mlops]**

Ir em **Actions → workflows**:

> "CI verde: ruff, mypy, bandit, pytest com 70% de cobertura, e docker build. Cobertura acima dos 60% obrigatórios.
>
> CD tem um approval gate manual antes de produção — human-in-the-loop, exigência de governança."

---

## SEGMENTO 4 — Resultados e Métricas (1 min 00 s)

**[Mostrar: `docs/MODEL_CARD.md` — abrir rapidamente]**

> "Resultados consolidados:
>
> **Modelo**: AUC 0.95, Precision 0.96, F1 0.84 no holdout temporal.
>
> **Qualidade do agente**: RAGAS faithfulness acima de 0.85, answer relevancy acima de 0.80 — avaliado contra 20 pares do golden set.
>
> **LLM-as-judge**: 3 critérios — precisão técnica, clareza para não-especialistas, e impacto na decisão de negócio. Esse terceiro critério é o mais importante: o analista consegue agir com base na resposta?
>
> **Segurança**: 5 cenários de red teaming testados — jailbreak, extração de dados de treino, prompt injection via RAG, escalação de privilégios, data leakage. Guardrails bloquearam 100% dos ataques testados."

---

## SEGMENTO 5 — Impacto de Negócio (1 min 00 s)

**[Voltar para o Grafana ou falar de frente]**

> "O que esse sistema entrega para o negócio?
>
> **Velocidade**: inferência em menos de 200ms. Um analista humano leva minutos por transação.
>
> **Explicabilidade**: o agente não diz só 'fraude' — diz *por que* é fraude, quais features contribuíram, e o que o analista deve fazer. Isso reduz fricção e aumenta confiança na automação.
>
> **Confiabilidade operacional**: drift detectado antes de degradar performance. Retreino automático quando PSI ultrapassa o threshold. Modelo novo só vai para produção após aprovação humana — governance by design.
>
> **Conformidade**: dados PCA anonimizados na origem, PII removida de inputs e outputs pelo Presidio, plano LGPD mapeando base legal, direitos dos titulares e fluxo de dados.
>
> O resultado é um sistema que um time de fraude real poderia operar: auditável, monitorado, seguro, e que explica suas decisões em linguagem que o analista entende."

---

## Fechamento (15 s)

> "Código aberto em GitHub. Toda a infraestrutura sobe com `make demo`.
> Prontos para perguntas."

---

## Perguntas Esperadas — Respostas Rápidas

### Técnicas (banca ML/Engenharia)

**"Por que threshold 0.25 e não 0.5?"**
> "O threshold foi otimizado automaticamente no treino para maximizar F1 no holdout. Com 0.17% de fraude, 0.5 seria muito conservador — muita fraude passaria. F1 balanceia precision e recall dado o custo assimétrico do erro."

**"Como o champion-challenger funciona?"**
> "Após cada retreino, o challenger precisa superar o champion em AUC por pelo menos 0.5% — `min_delta=0.005` em `src/models/train.py:263`. Abaixo disso, o champion é mantido. Se supera, vai para staging e só vai para produção após approval manual no GitHub Environment."

**"O que é temporal split e por que importa?"**
> "Separamos treino e teste por tempo — as transações mais antigas treinam, as mais recentes testam. Evita data leakage: num split aleatório, o modelo veria padrões futuros durante o treino, inflando AUC artificialmente. Em `src/models/baseline.py` na função `get_splits`."

**"RAGAS avalia o quê exatamente?"**
> "Quatro métricas: faithfulness (a resposta está ancorada no contexto recuperado?), answer relevancy (responde à pergunta?), context precision (o contexto recuperado é relevante?), context recall (toda informação necessária foi recuperada?). Contra 20 pares do golden set em `data/golden_set/`."

**"Como o guardrail detecta prompt injection?"**
> "Em `src/security/guardrails.py` — regex patterns para frases típicas de injection ('ignore previous instructions', 'you are now', etc.) + limite de 4096 caracteres. Output guardrail usa Presidio para remover CPF, email, telefone da resposta."

**"O MLP PyTorch treina mesmo?"**
> "Sim, mas Python 3.13 não tem suporte PyTorch — é ignorado automaticamente com `importorskip`. Em Python 3.12 ou 3.11, `make train` treina e registra o MLP no MLflow como `fraud_detector_mlp`."

### Negócio (banca executiva)

**"Quantas fraudes esse sistema teria capturado?"**
> "Com AUC 0.95 e precision 0.96, em 492 fraudes do dataset, o modelo teria bloqueado cerca de 420 com menos de 20 falsos positivos — ou seja, menos de 20 transações legítimas incorretamente bloqueadas para cada 420 fraudes detectadas."

**"Quanto custa operar isso?"**
> "Localmente, custo zero — Docker + sklearn. Em produção no Cloud Run, com scale-to-zero, o custo é proporcional ao volume. Dois milhões de requisições por mês entram no free tier do Cloud Run. O LLM com gpt-4o-mini custa ~$0.15/M tokens — uma consulta ao agente usa cerca de 500 tokens, ou $0.075 por mil consultas."

**"Por que usar LLM para explicar fraude? O modelo não já dá o score?"**
> "O score numérico não é acionável para um analista — ele precisa saber: *por que* esse score? O que fazer? O agente converte SHAP values em linguagem natural, busca casos similares, e dá uma recomendação concreta. Isso reduz o tempo de decisão do analista de minutos para segundos."

---

## Plano de Contingência

### Se a API não responder

```bash
# Terminal 2 — reiniciar API
MLFLOW_TRACKING_URI=http://localhost:5000 PYTHONPATH=. \
  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Se não subir em 30 segundos, mostrar o código diretamente:
- Abrir `src/serving/app.py` — mostrar o endpoint `/predict` (linha 211-239)
- Abrir `src/agent/tools.py` — mostrar o `fraud_predictor` tool (linha 26-87)

### Se o agente travar (LLM lento)

> "O agente está usando Ollama local — latência esperada de 10-20 segundos. Enquanto aguarda, posso mostrar como o ReAct funciona internamente..."

Abrir `src/agent/react_agent.py` e mostrar o loop Thought → Action → Observation.

### Se o Grafana não tiver dados

> "Os dados aparecem após algumas requisições à API. O dashboard foi configurado com 11 painéis — deixo o sistema gerar algumas requisições e os gráficos populam em tempo real."

Fazer 3-5 requisições rápidas via curl no terminal:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":9800,"Amount":850,"V14":-6.5,"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,"V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V15":0,"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,"V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,"V28":0}'
```

### Se nada funcionar — Demo offline

Mostrar nesta ordem:
1. `docs/ARCHITECTURE.md` — diagrama de arquitetura
2. `docs/MODEL_CARD.md` — métricas do modelo
3. `docs/OWASP_MAPPING.md` — segurança
4. `docs/RED_TEAM_REPORT.md` — red teaming
5. `.github/workflows/ci.yml` — CI/CD

---

## Guia de Tempo

```
00:00 — 01:00  Segmento 1: Problema
01:00 — 02:30  Segmento 2: Abordagem / Arquitetura
02:30 — 08:00  Segmento 3: Demo ao vivo
  02:30 — 03:15  3.1 MLflow Registry
  03:15 — 04:15  3.2 Predição via API
  04:15 — 05:45  3.3 Agente ReAct
  05:45 — 06:30  3.4 Langfuse
  06:30 — 07:30  3.5 Grafana
  07:30 — 08:00  3.6 CI/CD GitHub
08:00 — 09:00  Segmento 4: Resultados e Métricas
09:00 — 10:00  Segmento 5: Impacto de Negócio + Fechamento
```

**Sinal de atenção**: se você chegar no segmento 3.4 (Langfuse) com mais de 6 minutos marcados, pule 3.5 (Grafana) e vá direto para o CI/CD.

---

## Checklist Pre-Demo (5 min antes)

- [ ] `docker compose ps` — todos os serviços `Up`
- [ ] `curl http://localhost:8000/health` — `{"status":"ok","model_loaded":true}`
- [ ] Aba 1 (API docs) — carrega sem erro
- [ ] Aba 2 (MLflow) — modelo `fraud_detector_rf@Production` visível
- [ ] Aba 3 (Grafana) — dashboard abre, credenciais `admin/datathon`
- [ ] Aba 4 (Langfuse) — interface carrega
- [ ] Aba 5 (GitHub) — CI verde no branch `main`
- [ ] JSON de teste copiado na área de transferência
- [ ] Timer no celular configurado para 9 min 30 s (buffer de 30s)
