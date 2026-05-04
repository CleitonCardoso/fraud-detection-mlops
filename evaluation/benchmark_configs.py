"""Agent benchmark across multiple configurations — RAGAS + LLM-as-judge.

Evaluates 5 configurations varying temperature, retrieval k, chunk size,
and model to find the best trade-off between quality and latency.
Results are logged to MLflow for comparison and saved as CSV.

Usage:
    PYTHONPATH=. python evaluation/benchmark_configs.py
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "data/golden_set/golden_set.json"


@dataclass
class BenchmarkConfig:
    name: str
    description: str
    temperature: float
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int
    ollama_model: str


CONFIGS = [
    BenchmarkConfig(
        name="A_deterministic_baseline",
        description="Temperature=0, k=3, chunk=300 — deterministic baseline, standard retrieval",
        temperature=0.0,
        retrieval_k=3,
        chunk_size=300,
        chunk_overlap=30,
        ollama_model="llama3.2:1b",
    ),
    BenchmarkConfig(
        name="B_creative_larger_context",
        description="Temperature=0.3, k=5, chunk=300 — more creative answers, broader retrieval",
        temperature=0.3,
        retrieval_k=5,
        chunk_size=300,
        chunk_overlap=30,
        ollama_model="llama3.2:1b",
    ),
    BenchmarkConfig(
        name="C_larger_chunks",
        description="Temperature=0, k=3, chunk=512 — larger chunks preserve more context per document",
        temperature=0.0,
        retrieval_k=3,
        chunk_size=512,
        chunk_overlap=50,
        ollama_model="llama3.2:1b",
    ),
    BenchmarkConfig(
        name="D_high_recall_retrieval",
        description="Temperature=0, k=7, chunk=300 — maximizes context recall at cost of precision",
        temperature=0.0,
        retrieval_k=7,
        chunk_size=300,
        chunk_overlap=30,
        ollama_model="llama3.2:1b",
    ),
    BenchmarkConfig(
        name="E_larger_model",
        description="Temperature=0, k=3, chunk=300 — same as A but with llama3.2:3b for quality comparison",
        temperature=0.0,
        retrieval_k=3,
        chunk_size=300,
        chunk_overlap=30,
        ollama_model="llama3.2:3b",
    ),
]


def _get_llm(config: BenchmarkConfig):
    """Return LLM for config — Ollama if available, else OpenAI."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import urllib.request

        from langchain_ollama import ChatOllama
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        logger.info("LLM: Ollama %s (temperature=%.1f)", config.ollama_model, config.temperature)
        return ChatOllama(model=config.ollama_model, base_url=ollama_url, temperature=config.temperature)
    except Exception:
        pass

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        from langchain_openai import ChatOpenAI
        logger.info("LLM: OpenAI gpt-4o-mini (temperature=%.1f)", config.temperature)
        return ChatOpenAI(model="gpt-4o-mini", temperature=config.temperature, api_key=openai_key)  # type: ignore[arg-type]

    raise RuntimeError("Nenhum LLM disponível. Inicie Ollama ou configure OPENAI_API_KEY.")


def _get_embeddings():
    """Return embeddings backend — Ollama if available, else OpenAI."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import urllib.request

        from langchain_ollama import OllamaEmbeddings
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        return OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    except Exception:
        pass

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    raise RuntimeError("Nenhum backend de embeddings disponível.")


def _build_index_for_config(config: BenchmarkConfig):
    """Build a FAISS index with the config's chunk parameters."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from src.agent.rag_pipeline import FRAUD_KNOWLEDGE_BASE

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    docs = [Document(page_content=text) for text in FRAUD_KNOWLEDGE_BASE]
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, _get_embeddings())
    logger.info("Index: %d chunks (chunk_size=%d, overlap=%d)", len(chunks), config.chunk_size, config.chunk_overlap)
    return store


def _run_agent_with_config(config: BenchmarkConfig, question: str) -> tuple[str, float]:
    """Run the agent with a specific config. Returns (answer, latency_seconds)."""
    from langgraph.prebuilt import create_react_agent

    from src.agent.tools import drift_report, fraud_predictor, transaction_lookup

    SYSTEM_PROMPT = (
        "Você é um assistente especializado em detecção de fraude em transações financeiras. "
        "Responda sempre em português. Seja preciso, objetivo e baseie suas respostas nos dados disponíveis. "
        "Use as ferramentas disponíveis para responder perguntas sobre transações, drift e risco de fraude."
    )

    llm = _get_llm(config)
    agent = create_react_agent(llm, [fraud_predictor, transaction_lookup, drift_report], prompt=SYSTEM_PROMPT)

    start = time.perf_counter()
    result = agent.invoke({"messages": [("user", question)]})
    latency = time.perf_counter() - start

    messages = result.get("messages", [])
    answer = messages[-1].content if messages else ""
    return answer, latency


def run_benchmark(
    configs: list[BenchmarkConfig] = CONFIGS,
    golden_set_path: str = GOLDEN_SET_PATH,
) -> pd.DataFrame:
    """Run RAGAS evaluation for each config. Returns a ranked comparison DataFrame."""
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

    with open(golden_set_path) as f:
        golden_set = json.load(f)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))

    ragas_llm = LangchainLLMWrapper(_get_llm(configs[0]))
    ragas_emb = LangchainEmbeddingsWrapper(_get_embeddings())

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    summary_rows = []

    for config in configs:
        logger.info("=" * 60)
        logger.info("Config: %s", config.name)
        logger.info("%s", config.description)

        store = _build_index_for_config(config)
        rows = []
        total_latency = 0.0

        for i, item in enumerate(golden_set, 1):
            logger.info("[%d/%d] %s", i, len(golden_set), item["query"][:60])
            answer, latency = _run_agent_with_config(config, item["query"])
            contexts = [
                doc.page_content
                for doc in store.similarity_search(item["query"], k=config.retrieval_k)
            ]
            total_latency += latency
            rows.append({
                "question": item["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
            })

        dataset = Dataset.from_list(rows)
        scores = evaluate(  # type: ignore[call-overload]
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            raise_exceptions=False,
        )

        from typing import Any

        def _mean(v: Any) -> float:
            """RAGAS returns per-sample lists — take the mean, skip NaN."""
            import math
            vals = [x for x in (v if isinstance(v, list) else [v]) if x is not None and not math.isnan(x)]
            return float(sum(vals) / len(vals)) if vals else 0.0

        s: Any = scores
        metrics = {
            "faithfulness": _mean(s["faithfulness"]),
            "answer_relevancy": _mean(s["answer_relevancy"]),
            "context_precision": _mean(s["context_precision"]),
            "context_recall": _mean(s["context_recall"]),
            "avg_latency_s": round(total_latency / len(golden_set), 3),
            "ragas_composite": round(
                (_mean(s["faithfulness"]) +
                 _mean(s["answer_relevancy"]) +
                 _mean(s["context_precision"]) +
                 _mean(s["context_recall"])) / 4,
                4,
            ),
        }

        with mlflow.start_run(run_name=f"benchmark_{config.name}"):
            mlflow.log_params({
                "config_name": config.name,
                "temperature": config.temperature,
                "retrieval_k": config.retrieval_k,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "ollama_model": config.ollama_model,
            })
            mlflow.log_metrics(metrics)
            mlflow.set_tag("run_type", "benchmark")
            mlflow.set_tag("description", config.description)

        logger.info(
            "→ composite=%.4f  faithfulness=%.4f  relevancy=%.4f  latency=%.2fs",
            metrics["ragas_composite"],
            metrics["faithfulness"],
            metrics["answer_relevancy"],
            metrics["avg_latency_s"],
        )

        summary_rows.append({
            "config": config.name,
            "description": config.description,
            "temperature": config.temperature,
            "retrieval_k": config.retrieval_k,
            "chunk_size": config.chunk_size,
            "model": config.ollama_model,
            **metrics,
        })

    return pd.DataFrame(summary_rows).sort_values("ragas_composite", ascending=False)


def _print_summary(df: pd.DataFrame) -> None:
    cols = ["config", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_composite", "avg_latency_s"]
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS — sorted by RAGAS composite score")
    print("=" * 80)
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print("=" * 80)
    best = df.iloc[0]
    print(f"\nBest config : {best['config']}")
    print(f"Composite   : {best['ragas_composite']:.4f}")
    print(f"Avg latency : {best['avg_latency_s']:.2f}s")
    print(f"Description : {best['description']}")


if __name__ == "__main__":
    import sys
    import tempfile

    smoke = "--smoke" in sys.argv

    if smoke:
        logger.info("Running in SMOKE mode (2 configs × 2 questions)")
        with open(GOLDEN_SET_PATH) as _f:
            _full = json.load(_f)
        _tmp = Path(tempfile.mktemp(suffix=".json"))
        _tmp.write_text(json.dumps(_full[:2]))
        df = run_benchmark(configs=CONFIGS[:2], golden_set_path=str(_tmp))
        _tmp.unlink(missing_ok=True)
    else:
        df = run_benchmark()

    _print_summary(df)
    output_path = "data/processed/benchmark_results.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)
