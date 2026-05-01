"""PII detection and anonymization using Microsoft Presidio."""
import logging

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
)
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

_registry = RecognizerRegistry()
_registry.load_predefined_recognizers()

# BR recognizers registered for "en" so no spacy pt model is required
_registry.add_recognizer(PatternRecognizer(
    supported_entity="BR_CPF",
    patterns=[Pattern("BR_CPF", r"\d{3}\.\d{3}\.\d{3}-\d{2}", 0.85)],
    supported_language="en",
))
_registry.add_recognizer(PatternRecognizer(
    supported_entity="BR_CNPJ",
    patterns=[Pattern("BR_CNPJ", r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", 0.85)],
    supported_language="en",
))
_registry.add_recognizer(PatternRecognizer(
    supported_entity="BR_RG",
    patterns=[Pattern("BR_RG", r"\d{1,2}\.\d{3}\.\d{3}-[\dXx]", 0.65)],
    supported_language="en",
))

_analyzer = AnalyzerEngine(registry=_registry)
_anonymizer = AnonymizerEngine()

PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "BR_CPF",
    "BR_RG",
    "BR_CNPJ",
]


def anonymize(text: str, language: str = "en") -> str:
    """Detect and anonymize PII in a text string.

    Args:
        text: Input text that may contain PII.
        language: Language code for the analyzer.

    Returns:
        Text with PII replaced by entity type placeholders.
    """
    results = _analyzer.analyze(text=text, language=language, entities=PII_ENTITIES)
    if not results:
        return text
    logger.warning("PII detectado: %d entidade(s) — anonimizando", len(results))
    return _anonymizer.anonymize(text=text, analyzer_results=results).text


def contains_pii(text: str, language: str = "en") -> bool:
    """Check whether a text string contains any PII.

    Args:
        text: Text to inspect.
        language: Language code.

    Returns:
        True if PII is found.
    """
    results = _analyzer.analyze(text=text, language=language, entities=PII_ENTITIES)
    return len(results) > 0
