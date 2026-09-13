"""Prompt injection defenses — the content harness (task_4).

AgentMesh has structural defenses (egress filter, bounded loops, idempotency)
that limit what an injected prompt can DO. This module adds content defenses
that prevent injection from reaching the LLM and leaking the system prompt.

Three layers:
  1. Input sanitizer — regex-based check on user-supplied fields before
     they enter the prompt. Catches known patterns like "ignore previous
     instructions", "system:", "repeat your instructions". Flags, doesn't
     censor — the workflow decides whether to block.
  2. Output sanitizer — check on the LLM's response before it reaches the
     API. Catches system prompt leakage ("you are a sourcing agent",
     "my rules are:"). Redacts leaked content from the rationale.
  3. Injection monitoring — structured logging of every flagged attempt.

Honest limitations:
  - Novel injection patterns (rephrased attacks) are NOT caught by regex
  - Indirect injection (injection via tool outputs, not user input) is NOT
    covered by the input sanitizer
  - Model-level vulnerabilities (jailbreaks that don't match any pattern)
    are NOT solved by this module
  This is defense in depth, not defense in total.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("agentmesh.injection_defense")


class InjectionVerdict(str, Enum):
    CLEAN = "clean"
    FLAGGED = "flagged"
    BLOCKED = "blocked"


@dataclass
class SanitizationResult:
    verdict: InjectionVerdict = InjectionVerdict.CLEAN
    patterns_matched: list[str] = field(default_factory=list)
    original_text: str = ""
    sanitized_text: str = ""


# Known injection patterns — regex-based, case-insensitive
# These are NOT exhaustive. They catch the most common patterns from
# OWASP LLM Top 10 and public prompt injection examples.
INPUT_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "ignore_previous_instructions"),
    (r"ignore\s+(all\s+)?prior\s+instructions", "ignore_prior_instructions"),
    (r"disregard\s+(all\s+)?previous", "disregard_previous"),
    (r"forget\s+(all\s+)?previous\s+instructions", "forget_previous"),
    (r"\bsystem\s*:", "system_prefix"),
    (r"\bassistant\s*:", "assistant_prefix"),
    (r"pretend\s+(you\s+are|to\s+be)", "pretend_identity"),
    (r"act\s+as\s+(if\s+you\s+are|a)", "act_as"),
    (r"repeat\s+(all\s+)?(your\s+)?instructions", "repeat_instructions"),
    (r"(show|reveal|print|output|display)\s+.*?(your\s+)?(system\s+)?prompt", "reveal_prompt"),
    (r"(show|reveal|print|list)\s+(your\s+)?rules", "reveal_rules"),
    (r"for\s+educational\s+purposes", "educational_purpose"),
    (r"jailbreak", "jailbreak_keyword"),
    (r"override\s+(your\s+)?(safety|content)\s+(filter|policy)", "override_safety"),
    (r"you\s+are\s+now\s+(in\s+)?(developer|admin|root)\s+mode", "developer_mode"),
    (r"\[INST\]", "inst_tag"),
    (r"</s>", "end_of_text_tag"),
]

# Output patterns — system prompt leakage detection
OUTPUT_PATTERNS = [
    (r"you\s+are\s+a\s+(sourcing|hiring|agent)", "system_prompt_leak_role"),
    (r"my\s+(system\s+)?prompt\s+says", "system_prompt_reference"),
    (r"my\s+rules?\s+(are|is)\s*:", "rules_leak"),
    (r"my\s+instructions?\s+(are|is)\s*:", "instructions_leak"),
    (r"(here\s+are|these\s+are)\s+my\s+(system\s+)?(rules|instructions)", "rules_disclosure"),
]


def sanitize_input(text: str, block: bool = False) -> SanitizationResult:
    """Check user-supplied text for known injection patterns.

    Args:
        text: the user-supplied input text
        block: if True, flagged inputs get verdict=BLOCKED; if False, FLAGGED

    Returns:
        SanitizationResult with verdict, matched patterns, and original text.
        The text is NOT modified — the caller decides what to do with flagged input.
    """
    if not text:
        return SanitizationResult()

    matched = []
    text_lower = text.lower()

    for pattern, name in INPUT_PATTERNS:
        if re.search(pattern, text_lower):
            matched.append(name)

    if matched:
        verdict = InjectionVerdict.BLOCKED if block else InjectionVerdict.FLAGGED
        logger.warning(
            "INJECTION_ATTEMPT_FLAGGED patterns=%s verdict=%s text_preview=%.80s",
            matched, verdict.value, text[:80],
        )
        return SanitizationResult(
            verdict=verdict,
            patterns_matched=matched,
            original_text=text,
            sanitized_text=text,  # not modified — caller decides
        )

    return SanitizationResult(original_text=text, sanitized_text=text)


def sanitize_output(text: str) -> SanitizationResult:
    """Check LLM output for system prompt leakage.

    If leakage is detected, the leaked content is redacted from the
    sanitized_text (replaced with [REDACTED]).

    Returns:
        SanitizationResult with verdict, matched patterns, and sanitized text
        (leaked content redacted if found).
    """
    if not text:
        return SanitizationResult()

    matched = []
    sanitized = text

    for pattern, name in OUTPUT_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            matched.append(name)
            # Redact the matched portion and the rest of the line
            for m in matches:
                start = m.start()
                # Find end of line
                line_end = text.find("\n", start)
                if line_end == -1:
                    line_end = len(text)
                sanitized = sanitized[:start] + "[REDACTED]" + sanitized[line_end:]

    if matched:
        logger.warning(
            "SYSTEM_PROMPT_LEAK_DETECTED patterns=%s redacted=true",
            matched,
        )
        return SanitizationResult(
            verdict=InjectionVerdict.FLAGGED,
            patterns_matched=matched,
            original_text=text,
            sanitized_text=sanitized,
        )

    return SanitizationResult(original_text=text, sanitized_text=text)


def sanitize_brief_fields(brief_dict: dict, string_fields: list[str], block: bool = False) -> dict:
    """Sanitize all string fields in a brief/input dict.

    Returns the brief dict with a _injection_flags key containing the
    sanitization results for each field. Does NOT modify the original
    field values — the workflow decides whether to block.
    """
    flags = {}
    for field_name in string_fields:
        value = brief_dict.get(field_name, "")
        if isinstance(value, str) and value:
            result = sanitize_input(value, block=block)
            if result.verdict != InjectionVerdict.CLEAN:
                flags[field_name] = {
                    "verdict": result.verdict.value,
                    "patterns": result.patterns_matched,
                }

    brief_dict["_injection_flags"] = flags
    return brief_dict
