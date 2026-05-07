"""RAG pipeline: FAISS vector store over fraud knowledge base."""
import logging
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

INDEX_PATH = "data/processed/faiss_index"

FRAUD_KNOWLEDGE_BASE = [
    "Transações realizadas em horário incomum (00h–05h) têm probabilidade três vezes maior de serem fraudulentas.",
    "Valores acima de R$1.000 combinados com feature V14 menor que -5 são forte indicativo de fraude.",
    "Fraudes em cartão de crédito frequentemente ocorrem em sequência rápida de pequenas transações antes de uma grande.",
    "O feature V17 negativo associado a V12 positivo é padrão reconhecido em fraudes de card-not-present.",
    "Transações com Amount igual a valores redondos (100, 500, 1000) merecem atenção adicional.",
    "PSI acima de 0.2 indica que a distribuição atual dos dados divergiu significativamente do período de treino.",
    "O modelo Random Forest atingiu AUC de 0.95 no conjunto de teste com threshold de 0.5.",
    "Class imbalance: 99.83% de transações legítimas e 0.17% de fraudes no dataset de treino.",
    "LGPD: features V1-V28 são componentes PCA de dados originais — nenhuma informação pessoal identificável está presente.",
    "Champion-challenger: um novo modelo só é promovido a produção se superar o champion atual em pelo menos 0.5% de AUC.",
    "Drift de dados é monitorado diariamente via Evidently com PSI. Threshold de warning em 0.1, retrain em 0.2.",
    "O agente ReAct possui três ferramentas: fraud_predictor, transaction_lookup e drift_report.",
    "Precision mede quantas das transações classificadas como fraude são realmente fraude.",
    "Recall mede quantas das fraudes reais foram detectadas — mais crítico do que precision neste domínio.",
    "F1-score é a média harmônica entre precision e recall, equilibrando os dois objetivos.",
    "O threshold padrão de classificação é 0.5. Scores acima disso são marcados como fraude.",
    "O sistema usa SHAP para explicar as predições, mostrando as features que mais contribuíram para cada decisão.",
    "Retreino automático é disparado quando PSI > 0.2 em qualquer feature monitorada.",
    "O deploy em produção requer aprovação humana via GitHub Environment gate — human-in-the-loop.",
    "Langfuse monitora cada chamada do agente: tokens, latência, tools usadas, inputs e outputs.",
]


def _get_embeddings():
    """Return embeddings backend: Ollama if available, else OpenAI."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    try:
        import urllib.request

        from langchain_ollama import OllamaEmbeddings
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        logger.info("Usando Ollama embeddings (nomic-embed-text)")
        return OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    except Exception:
        pass

    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        logger.info("Usando OpenAI embeddings")
        return OpenAIEmbeddings(model="text-embedding-3-small")

    raise RuntimeError(
        "Nenhum backend de embeddings disponível. "
        "Inicie o Ollama (ollama serve) ou configure OPENAI_API_KEY."
    )


def build_index() -> FAISS:
    """Build and persist FAISS index from the fraud knowledge base."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = [Document(page_content=text) for text in FRAUD_KNOWLEDGE_BASE]
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    store.save_local(INDEX_PATH)
    logger.info("FAISS index criado com %d chunks em %s", len(chunks), INDEX_PATH)
    return store


def load_index() -> FAISS:
    """Load FAISS index from disk, building it if it does not exist."""
    embeddings = _get_embeddings()
    if Path(INDEX_PATH).exists():
        logger.info("Carregando FAISS index de %s", INDEX_PATH)
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index não encontrado — construindo...")
    return build_index()


_INDEX_CACHE: FAISS | None = None


def retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieve top-k relevant chunks for a query."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = load_index()
    docs = _INDEX_CACHE.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
