# ADR-002 — LangGraph para implementação do agente ReAct

**Status:** Aceito  
**Data:** 2026-04-29  
**Autores:** Cleiton Cardoso

---

## Contexto

Precisamos de um agente LLM capaz de raciocinar sobre transações suspeitas, consultar a base de conhecimento via RAG e reportar o estado do drift — tudo em ciclos Thought-Action-Observation.

## Decisão

Usar **LangGraph** com `create_react_agent` ao invés de `langchain.agents.AgentExecutor`.

## Justificativa

- LangChain 1.x deprecou `AgentExecutor` e `create_react_agent` do pacote principal em favor de LangGraph.
- LangGraph oferece controle de estado explícito via grafo dirigido, facilitando debugging.
- `langgraph.prebuilt.create_react_agent` é a API oficial recomendada para agentes ReAct em 2025.
- Compatível com qualquer `ChatModel` (OpenAI, Ollama, Anthropic) sem mudança de código.

## Consequências

- **Positivo:** Código alinhado com a direção oficial do ecossistema LangChain/LangGraph.
- **Positivo:** Fallback automático entre OpenAI e Ollama sem alterar a lógica do agente.
- **Negativo:** LangGraph adiciona uma dependência de grafo que pode ser overkill para agentes simples.
- **Negativo:** Modelos pequenos (< 3B parâmetros) têm dificuldade com tool calling confiável.
