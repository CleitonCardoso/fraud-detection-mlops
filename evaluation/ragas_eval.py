"""RAGAS evaluation of the RAG pipeline against the golden set."""
import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.agent.rag_pipeline import retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "data/golden_set/golden_set.json"


def run_ragas(golden_set_path: str = GOLDEN_SET_PATH) -> dict[str, float]:
    """Evaluate the RAG pipeline using RAGAS 4 metrics.

    Args:
        golden_set_path: Path to the golden set JSON file.

    Returns:
        Dictionary with faithfulness, answer_relevancy, context_precision, context_recall.
    """
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    logger.info("Avaliando %d pares do golden set", len(golden_set))

    from src.agent.react_agent import query as agent_query

    rows = []
    for item in golden_set:
        result = agent_query(item["query"])
        contexts = retrieve(item["query"], k=3)
        rows.append({
            "question": item["query"],
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": item["expected_answer"],
        })

    dataset = Dataset.from_list(rows)
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

    import mlflow, os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_name="ragas_evaluation"):
        mlflow.log_metrics(metrics)

    logger.info("RAGAS scores: %s", metrics)
    return metrics


if __name__ == "__main__":
    results = run_ragas()
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
