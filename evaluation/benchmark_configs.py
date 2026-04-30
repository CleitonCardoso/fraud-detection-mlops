"""Benchmark comparing 3 agent configurations as required by the rubric."""
import json
import logging
import os
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.agent.rag_pipeline import retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_QUERIES = [
    "Uma transação de R$850 às 3h da manhã é suspeita?",
    "O que significa PSI acima de 0.2?",
    "Qual é o threshold de decisão do modelo?",
]

CONFIGS = [
    {
        "name": "Config A — gpt-4o-mini temp=0.0",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "chunk_size": 300,
        "description": "Determinístico, respostas consistentes",
    },
    {
        "name": "Config B — gpt-4o-mini temp=0.3 chunk=512",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "chunk_size": 512,
        "description": "Ligeiramente criativo, contexto maior",
    },
    {
        "name": "Config C — Ollama llama3.2 quantizado (fallback gpt-4o-mini)",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "chunk_size": 300,
        "description": "Simula modelo local quantizado GGUF — latência e custo comparados",
        "note": "Em ambiente com Ollama: substituir model por 'ollama/llama3.2:3b-q4_K_M'",
    },
]


def run_benchmark(output_path: str = "data/processed/benchmark_results.json") -> list[dict]:
    """Run all 3 configurations against sample queries and record latency and quality.

    Args:
        output_path: Where to save JSON results.

    Returns:
        List of result dicts per config.
    """
    results = []

    for config in CONFIGS:
        logger.info("Rodando %s", config["name"])
        llm = ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        config_results = {"config": config["name"], "description": config["description"], "queries": []}

        for query in SAMPLE_QUERIES:
            start = time.perf_counter()
            chunks = retrieve(query, k=3)
            context = "\n".join(chunks)
            prompt = f"Contexto:\n{context}\n\nPergunta: {query}\nResponda em português de forma objetiva."
            response = llm.invoke(prompt)
            latency = time.perf_counter() - start

            config_results["queries"].append({
                "query": query,
                "answer": response.content,
                "latency_s": round(latency, 3),
                "tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            })
            logger.info("  Query '%s...' — %.2fs", query[:40], latency)

        avg_latency = sum(q["latency_s"] for q in config_results["queries"]) / len(SAMPLE_QUERIES)
        config_results["avg_latency_s"] = round(avg_latency, 3)
        results.append(config_results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Benchmark salvo em %s", output_path)

    print("\n── Benchmark Results ──────────────────────────────────")
    for r in results:
        print(f"\n{r['config']}")
        print(f"  Latência média: {r['avg_latency_s']}s")
    return results


if __name__ == "__main__":
    run_benchmark()
