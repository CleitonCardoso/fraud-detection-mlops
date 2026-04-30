"""Input and output guardrails for the fraud detection agent."""
import logging
import re

from src.security.pii_detection import anonymize, contains_pii

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s*:",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"forget\s+(everything|all|your\s+instructions)",
    r"disregard\s+(all\s+)?previous",
    r"bypass\s+(your\s+)?(safety|guidelines|instructions)",
    r"jailbreak",
    r"do\s+anything\s+now",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

MAX_INPUT_LENGTH = 4096


class InputGuardrail:
    """Validates and sanitizes user input before sending to the LLM."""

    def validate(self, text: str) -> tuple[bool, str]:
        """Validate user input against injection and size constraints.

        Args:
            text: Raw user input.

        Returns:
            Tuple (is_valid, reason). reason is 'OK' when valid.
        """
        if len(text) > MAX_INPUT_LENGTH:
            logger.warning("Input bloqueado: tamanho %d > %d", len(text), MAX_INPUT_LENGTH)
            return False, f"Input excede o limite de {MAX_INPUT_LENGTH} caracteres."

        for pattern in _COMPILED:
            if pattern.search(text):
                logger.warning("Prompt injection detectado: %.80s", text)
                return False, "Input bloqueado: padrão suspeito detectado."

        return True, "OK"

    def sanitize(self, text: str) -> str:
        """Anonymize PII from user input before forwarding to the LLM.

        Args:
            text: User input after validation.

        Returns:
            Text with PII replaced.
        """
        return anonymize(text)


class OutputGuardrail:
    """Validates and sanitizes LLM output before returning to the user."""

    def sanitize(self, text: str) -> str:
        """Remove PII from LLM output.

        Args:
            text: Raw LLM response.

        Returns:
            Sanitized response safe to return to the user.
        """
        if contains_pii(text):
            logger.warning("PII detectado no output do LLM — anonimizando")
            return anonymize(text)
        return text
