# OWASP Top 10 for LLM Applications — Mapeamento e Mitigações

> Referência: OWASP Top 10 for Large Language Model Applications (2025)

---

## LLM01 — Prompt Injection

**Descrição**: Usuário manipula o prompt para alterar o comportamento do LLM, ignorar instruções ou extrair informações internas.

**Vetores no nosso sistema**:
- Query direta ao `/agent/query` com instruções adversariais
- Dados maliciosos injetados via contexto RAG

**Mitigações implementadas**:
- `InputGuardrail` com 9 regex patterns detectando injection clássico
- Bloqueio com HTTP 400 e log de auditoria
- Prompt template fixo com separação clara entre instrução do sistema e input do usuário
- RAG: apenas chunks da nossa base de conhecimento controlada são injetados

**Status**: ✅ Mitigado

---

## LLM02 — Insecure Output Handling

**Descrição**: Output do LLM é processado sem sanitização, permitindo XSS, execução de código, etc.

**Vetores no nosso sistema**:
- Output do agente retornado diretamente sem filtragem

**Mitigações implementadas**:
- `OutputGuardrail` processa toda resposta antes de retornar ao usuário
- Presidio detecta e anonimiza PII no output
- FastAPI retorna JSON estruturado — sem renderização de HTML que permitiria XSS
- Logging do output para auditoria posterior

**Status**: ✅ Mitigado

---

## LLM06 — Sensitive Information Disclosure

**Descrição**: LLM revela dados sensíveis presentes no contexto de treinamento ou no prompt.

**Vetores no nosso sistema**:
- Perguntas indiretas sobre dados de treino ("quais CPFs estão no dataset?")
- Extração de informações do system prompt via perguntas criativas

**Mitigações implementadas**:
- Dataset usa features PCA — nenhum dado pessoal identificável presente
- OutputGuardrail com Presidio remove CPF, CNPJ, email, telefone do output
- Red team cenário RT5 testa exatamente este vetor (ver `RED_TEAM_REPORT.md`)
- System prompt não contém informações sensíveis

**Status**: ✅ Mitigado

---

## LLM08 — Excessive Agency

**Descrição**: Agente LLM recebe permissões ou capacidades além do necessário, podendo tomar ações não autorizadas.

**Vetores no nosso sistema**:
- Agente com tools que poderiam ser usadas para ações destrutivas
- Tool `fraud_predictor` com acesso ao MLflow

**Mitigações implementadas**:
- Tools limitadas a 3 ações read-only: predição, consulta RAG, status de drift
- Nenhuma tool permite escrita em banco de dados, envio de mensagens ou execução de código arbitrário
- `max_iterations=10` no AgentExecutor previne loops infinitos
- `handle_parsing_errors=True` previne quebra por output malformado

**Status**: ✅ Mitigado

---

## LLM09 — Misinformation

**Descrição**: LLM gera informações incorretas ou alucinatórias com alta confiança.

**Vetores no nosso sistema**:
- Agente respondendo sobre padrões de fraude sem base nos dados reais
- Scores ou métricas inventados

**Mitigações implementadas**:
- RAG garante que respostas se baseiam na base de conhecimento verificada
- RAGAS `faithfulness` mede proporção de claims suportados pelo contexto (alvo ≥ 0.85)
- LLM-as-judge avalia precisão técnica de cada resposta do golden set
- Tool `fraud_predictor` retorna score real do modelo — não gerado pelo LLM

**Status**: ✅ Mitigado com monitoramento contínuo
