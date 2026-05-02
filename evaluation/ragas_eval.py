"""RAGAS evaluation of the RAG pipeline against the golden set."""
import json
import logging
import os

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "data/golden_set/golden_set.json"


def _get_ragas_llm():
    """Return LangChain LLM for RAGAS: Ollama if available, else OpenAI."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import urllib.request

        from langchain_ollama import ChatOllama
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        logger.info("RAGAS usando Ollama LLM: %s", model)
        return ChatOllama(model=model, base_url=ollama_url, temperature=0.0)
    except Exception:
        pass

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai_key)

    raise RuntimeError("Nenhum LLM disponível para RAGAS.")


def _get_ragas_embeddings():
    """Return embeddings for RAGAS: Ollama if available, else OpenAI."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import urllib.request

        from langchain_ollama import OllamaEmbeddings
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        logger.info("RAGAS usando Ollama embeddings")
        return OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    except Exception:
        pass

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    raise RuntimeError("Nenhum backend de embeddings disponível para RAGAS.")


def run_ragas(golden_set_path: str = GOLDEN_SET_PATH) -> dict[str, float]:
    """Evaluate the RAG pipeline using RAGAS 4 metrics."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.agent.rag_pipeline import retrieve
    from src.agent.react_agent import query as agent_query

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    logger.info("Avaliando %d pares do golden set com RAGAS", len(golden_set))

    ragas_llm = LangchainLLMWrapper(_get_ragas_llm())
    ragas_emb = LangchainEmbeddingsWrapper(_get_ragas_embeddings())

    rows = []
    for i, item in enumerate(golden_set, 1):
        logger.info("[%d/%d] %s", i, len(golden_set), item["query"][:60])
        result = agent_query(item["query"])
        contexts = retrieve(item["query"], k=3)
        rows.append({
            "question": item["query"],
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
        })

    dataset = Dataset.from_list(rows)

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_name="ragas_evaluation"):
        mlflow.log_metrics(metrics)

    logger.info("RAGAS scores: %s", metrics)
    return metrics


if __name__ == "__main__":
    results = run_ragas()
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
