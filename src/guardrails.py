"""Input/output guardrails: validation, prompt-injection heuristics, confidence scoring."""
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.config import settings

# Heuristic patterns for prompt-injection / jailbreak attempts in the user query.
# Not a security boundary on its own -- paired with a strict system prompt that
# tells the LLM to treat retrieved context and the user query as data, not
# instructions.
_INJECTION_PATTERNS = [
    r"ignore (all|the|any) (previous|prior|above) instructions",
    r"disregard (all|the|any) (previous|prior|above)",
    r"you are now",
    r"act as (a|an)\s",
    r"reveal (your|the) (system|hidden) prompt",
    r"print (your|the) (system|initial) prompt",
    r"what (is|are) your instructions",
    r"jailbreak",
    r"pretend (you|to) (are|be)",
    r"developer mode",
    r"do anything now",
    r"\bDAN\b",
    r"override (your|the) rules",
    r"forget (everything|all) (you|above)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_SRC = r"(?:the |the provided |the given )?(?:document|ebook|context|pdf|source material)"

_REFUSAL_PATTERNS = [
    r"i don'?t have enough information",
    r"i (cannot|can'?t|am unable to) answer",
    rf"not covered (in|by) {_SRC}",
    rf"{_SRC} does(n'?t| not) (contain|mention|provide|include|address|cover|discuss)",
    rf"no information (about|regarding|on) this (is )?(available|provided) in {_SRC}",
    rf"cannot be answered (based on|from|using) {_SRC}",
    r"i don'?t know",
    rf"outside (the )?scope of {_SRC}",
    rf"not (mentioned|discussed|addressed|described|found) in {_SRC}",
    r"no mention of (this|that|it|such)",
    rf"is not (part of|included in|available in) {_SRC}",
    rf"(this|that) information is not (in|available in|part of) {_SRC}",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    sanitized_query: Optional[str] = None
    flagged_injection: bool = False


def validate_query(query: str) -> ValidationResult:
    """Validate + sanitize a raw user query before it touches retrieval/LLM."""
    if query is None:
        return ValidationResult(False, "Query is missing.")

    # strip control characters, normalize whitespace
    cleaned = _CONTROL_CHARS_RE.sub("", query).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if not cleaned or len(cleaned) < settings.MIN_QUERY_CHARS:
        return ValidationResult(False, "Query is empty or too short.")

    if len(cleaned) > settings.MAX_QUERY_CHARS:
        return ValidationResult(
            False,
            f"Query is too long ({len(cleaned)} chars). Max is {settings.MAX_QUERY_CHARS}.",
        )

    if _INJECTION_RE.search(cleaned):
        return ValidationResult(
            is_valid=False,
            reason="Query looks like a prompt-injection / instruction-override attempt.",
            sanitized_query=cleaned,
            flagged_injection=True,
        )

    return ValidationResult(True, sanitized_query=cleaned)


def compute_retrieval_confidence(matches: List[Dict]) -> float:
    """Map retrieval similarity scores to a 0-1 confidence estimate."""
    if not matches:
        return 0.0
    top = matches[0]["score"]
    avg_top3 = sum(m["score"] for m in matches[:3]) / min(3, len(matches))
    # weight the single best match higher, but reward consistent support
    confidence = 0.7 * top + 0.3 * avg_top3
    return round(max(0.0, min(1.0, confidence)), 3)


def passes_relevance_gate(matches: List[Dict]) -> bool:
    if not matches:
        return False
    return matches[0]["score"] >= settings.RELEVANCE_SCORE_THRESHOLD

def passes_groundedness_gate(groundedness: float) -> bool:
    return groundedness >= settings.GROUNDEDNESS_THRESHOLD

def looks_like_refusal(answer: str) -> bool:
    """Heuristic: does the model's own answer text amount to a refusal /
    'not in the document' response? Not airtight, but catches the common
    phrasings a well-behaved model uses when it follows the system prompt's
    instruction to say so plainly instead of guessing."""
    if not answer:
        return False
    return bool(_REFUSAL_RE.search(answer))


def sanitize_llm_output(text: str) -> str:
    """Strip anything that looks like a leaked system prompt / instruction block,
    and strip <think>...</think> reasoning blocks some models emit inline."""
    if not text:
        return text
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?i)system prompt:.*", "", text)
    return text.strip()
