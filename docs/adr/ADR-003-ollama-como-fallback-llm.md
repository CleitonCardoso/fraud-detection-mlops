# ADR-003 — Ollama como fallback local para OpenAI

**Status:** Aceito  
**Data:** 2026-04-30  
**Autores:** Cleiton Cardoso

---

## Contexto

O agente e o pipeline de avaliação (RAGAS, LLM-judge) dependem de um LLM. Em ambientes sem acesso à internet ou sem chave OpenAI, o sistema ficaria completamente inoperante.

## Decisão

Implementar detecção automática de backend LLM: **Ollama tem prioridade**, OpenAI é fallback quando Ollama não está disponível.

```python
# Ordem de prioridade em _get_llm():
# 1. Ollama (http://localhost:11434) — local, gratuito, sem dados enviados externamente
# 2. OpenAI (OPENAI_API_KEY) — nuvem, pago, melhor qualidade
# 3. RuntimeError se nenhum disponível
```

Modelos Ollama usados:
- **Chat/ReAct:** `llama3.2:3b` (melhor tool calling) ou `llama3.2:1b` (mais rápido)
- **Embeddings RAG:** `nomic-embed-text` (768 dimensões, compatível com FAISS)

## Justificativa

- Elimina dependência de credenciais externas para desenvolvimento e demo.
- Nenhum dado de transação (mesmo sintético) sai da máquina local — alinhado com LGPD.
- `nomic-embed-text` produz embeddings de qualidade comparável ao `text-embedding-3-small` para português.

## Consequências

- **Positivo:** Sistema funciona 100% offline e sem custo.
- **Positivo:** Conformidade LGPD mais fácil de demonstrar (dados não saem da máquina).
- **Negativo:** Modelos locais menores têm qualidade de raciocínio inferior ao GPT-4o.
- **Negativo:** Latência maior localmente (2–15s vs ~1s OpenAI).
- **Mitigação:** A variável `OLLAMA_MODEL` permite trocar o modelo sem alterar código.
