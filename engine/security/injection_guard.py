"""
Security layer, prompt injection detection.

Sits between the gRPC boundary and the LLM call.
Every prompt variant passes through this guard before execution.
A blocked prompt never reaches the model - the request fails fast
with a clear audit record of what was attempted.

ISO 27001 mapping:
  A.12.6 - Technical vulnerability management
  A.14.2 - Security in development and support processes
"""
import re
from utils.create_logger import get_logger

logger = get_logger(__name__)

# Patterns that indicate prompt injection attempts.
# Based on OWASP Top 10 for LLMs. LLM01: Prompt Injection.

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"act\s+as\s+(a|an|the)\s+",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"(do\s+anything\s+now|DAN)",
    r"jailbreak",
    r"system\s*prompt\s*:",
    r"<\s*system\s*>",
    r"\[INST\]",                    # Llama instruction injection
    r"###\s*instruction",           # Common prompt boundary exploit
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# PII patterns - inputs matching these are flagged for review.
# In a full ISO 27001 implementation these would trigger data
# classification escalation, not just a log warning.
_PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
    r"\b\d{16}\b",                        # Credit card
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
]

_PII_COMPILED = [re.compile(p) for p in _PII_PATTERNS]


class InjectionDetected(Exception):
    """Raised when a prompt variant contains injection patterns."""
    pass


def scan(text: str, trace_id: str, field: str = "input") -> str:
    """
    Scans a single text field for injection patterns and PII.

    trace_id: the request trace ID for audit logging.
    field: which field is being scanned ("variant_a", "variant_b", "input").

    Returns the original text unchanged if clean.
    Raises InjectionDetected if an injection pattern is found.
    Logs a warning (but does not block) if PII is detected.
    """
    for pattern in _COMPILED:
        if pattern.search(text):
            logger.warning(
                f"trace={trace_id} field={field} "
                f"event=injection_detected pattern={pattern.pattern!r}"
            )
            raise InjectionDetected(
                f"Prompt injection pattern detected in field '{field}'. "
                f"Request blocked. trace_id={trace_id}"
            )

    for pattern in _PII_COMPILED:
        if pattern.search(text):
            logger.warning(
                f"trace={trace_id} field={field} "
                f"event=pii_detected - data classification review required"
            )
            # PII in a prompt is a policy violation but not an injection attempt.
            # Log it and continue - the operator must review the audit log.
            break

    return text


def scan_request(
    trace_id: str,
    input_text: str,
    variant_a: str,
    variant_b: str,
) -> None:
    """
    Scans all three user-controlled fields of an EvaluateRequest.
    Raises InjectionDetected if any field fails.
    Called once per request before any LLM interaction begins.
    """
    scan(input_text, trace_id, field="input")
    scan(variant_a, trace_id, field="variant_a")
    scan(variant_b, trace_id, field="variant_b")