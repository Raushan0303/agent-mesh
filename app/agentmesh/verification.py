"""Tri-state verification — the third state that prevents duplicate side effects.

When verification itself fails (DB connection error, timeout, query
error), we can't determine if the side effect happened. This is UNKNOWN
— not FAILED. The wrong response to UNKNOWN is to retry the side effect
(which could cause a duplicate charge). The right response is to
escalate with a recovery record.

States:
  VERIFIED  — the side effect was confirmed (DB row exists, fields match)
  FAILED    — the side effect definitely didn't happen (DB row not found)
  UNKNOWN   — we can't tell (verification query failed, DB unreachable)

The critical distinction: FAILED means "safe to retry." UNKNOWN means
"NOT safe to retry — the side effect may have happened, we just can't
confirm it. Escalate to a human."
"""

from enum import Enum


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNKNOWN = "unknown"


def verification_result(
    status: VerificationStatus,
    *,
    side_effect_id: str = "",
    mismatches: list[str] | None = None,
    error: str = "",
    **extra,
) -> dict:
    """Build a tri-state verification result dict.

    All verification activities return this shape so workflows can
    switch on status without guessing field names.
    """
    result = {
        "status": status.value,
        "verified": status == VerificationStatus.VERIFIED,
        "side_effect_id": side_effect_id,
        "mismatches": mismatches or [],
    }
    if error:
        result["error"] = error
    result.update(extra)
    return result
