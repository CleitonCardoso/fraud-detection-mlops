"""LLM-as-judge evaluation with 3 criteria including a business criterion."""
import json
import logging
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "data/golden_set/golden_set.json"

JUDGE_PROMPT = """Você é um avaliador especializado em sistemas de detecção de fraude financeira.
Avalie a resposta abaixo em três critérios, pontuando de 1 a 5 cada um.

Pergunta: {question}
Resposta do sistema: {answer}
Resposta esperada: {expected}

Critérios de avaliação:
1. PRECISÃO TÉCNICA (1-5): A resposta usa corretamente termos e conceitos de detecção de fraude (PSI, AUC, features, drift)?
2. CLAREZA PARA NÃO-ESPECIALISTAS (1-5): Um gerente sem background técnico conseguiria entender e usar essa informação?
3. IMPACTO NA DECISÃO DE NEGÓCIO (1-5): A resposta fornece informação acionável que ajudaria um analista de fraude a tomar uma decisão concreta?

Responda APENAS com JSON válido no formato:
{{"precisao_tecnica": <1-5>, "clareza": <1-5>, "impacto_negocio": <1-5>, "justificativa": "<uma frase>"}}"""


def judge_answer(question: str, answer: str, expected: str, llm: ChatOpenAI) -> dict:
    """Evaluate a single Q&A pair using the LLM judge.

    Args:
        question: The original question.
        answer: The system's answer.
        expected: The expected answer from the golden set.
        llm: Configured ChatOpenAI instance.

    Returns:
        Dictionary with scores for each criterion.
    """
    prompt = JUDGE_PROMPT.format(question=question, answer=answer, expected=expected)
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        logger.warning("LLM judge retornou JSON inválido — usando zeros")
        return {"precisao_tecnica": 0, "clareza": 0, "impacto_negocio": 0, "justificativa": "parse error"}


def run_llm_judge(golden_set_path: str = GOLDEN_SET_PATH) -> dict[str, float]:
    """Run LLM-as-judge over the full golden set.

    Args:
        golden_set_path: Path to golden set JSON.

    Returns:
        Average scores per criterion across all golden set items.
    """
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

    from src.agent.react_agent import query as agent_query

    totals: dict[str, float] = {"precisao_tecnica": 0.0, "clareza": 0.0, "impacto_negocio": 0.0}

    for item in golden_set:
        result = agent_query(item["query"])
        scores = judge_answer(item["query"], result["answer"], item["expected_answer"], llm)
        for k in totals:
            totals[k] += scores.get(k, 0)
        logger.info("Query: %s | scores: %s", item["query"][:50], scores)

    n = len(golden_set)
    averages = {k: round(v / n, 3) for k, v in totals.items()}

    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_name="llm_judge_evaluation"):
        mlflow.log_metrics({f"judge_{k}": v for k, v in averages.items()})

    logger.info("LLM judge averages: %s", averages)
    return averages


if __name__ == "__main__":
    results = run_llm_judge()
    for k, v in results.items():
        print(f"{k}: {v:.3f} / 5.0")
