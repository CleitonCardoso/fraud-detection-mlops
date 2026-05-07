"""ReAct agent for fraud analysis using LangGraph + Langfuse telemetry."""
import logging
import os

from langgraph.prebuilt import create_react_agent

from src.agent.tools import drift_report, fraud_predictor, transaction_lookup

logger = logging.getLogger(__name__)

TOOLS = [fraud_predictor, transaction_lookup, drift_report]

SYSTEM_PROMPT = (
    "Você é um assistente especializado em detecção de fraude em transações financeiras. "
    "Responda sempre em português. Seja preciso, objetivo e baseie suas respostas nos dados disponíveis. "
    "Use as ferramentas disponíveis para responder perguntas sobre transações, drift e risco de fraude."
)


def _get_llm(model_name: str, temperature: float = 0.0):
    """Return LLM backend: Ollama if available, else OpenAI."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    try:
        import urllib.request

        from langchain_ollama import ChatOllama
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        logger.info("Usando Ollama LLM: %s", ollama_model)
        return ChatOllama(model=ollama_model, base_url=ollama_url, temperature=temperature)
    except Exception:
        pass

    if openai_key:
        from langchain_openai import ChatOpenAI
        logger.info("Usando OpenAI LLM: %s", model_name)
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_key)  # type: ignore[arg-type]

    raise RuntimeError(
        "Nenhum LLM disponível. Inicie o Ollama (ollama serve) ou configure OPENAI_API_KEY."
    )


def _langfuse_handler():
    """Return a Langfuse callback handler if credentials are configured, else None."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3001")

    if not public_key or not secret_key:
        logger.debug("Langfuse não configurado — LANGFUSE_PUBLIC_KEY/SECRET_KEY ausentes")
        return None

    try:
        from langfuse.callback import CallbackHandler
        handler = CallbackHandler(public_key=public_key, secret_key=secret_key, host=host)
        logger.info("Langfuse telemetria ativa em %s", host)
        return handler
    except ImportError:
        logger.warning("langfuse não instalado — sem telemetria LLM")
        return None


_AGENT_CACHE: dict = {}


def build_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Build the ReAct agent graph."""
    llm = _get_llm(model_name, temperature)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def query(question: str, model_name: str = "gpt-4o-mini") -> dict:
    """Run a question through the fraud detection agent."""
    if model_name not in _AGENT_CACHE:
        _AGENT_CACHE[model_name] = build_agent(model_name=model_name)
    agent = _AGENT_CACHE[model_name]

    callbacks = []
    handler = _langfuse_handler()
    if handler:
        callbacks.append(handler)

    config = {"callbacks": callbacks} if callbacks else {}
    result = agent.invoke({"messages": [("user", question)]}, config=config)

    messages = result.get("messages", [])
    answer = messages[-1].content if messages else ""
    tool_calls = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)

    logger.info("Agent query concluída: %d mensagens, %d tool calls", len(messages), tool_calls)
    return {"answer": answer, "steps": tool_calls}
