"""Mock applicant-tracking-system data — the hiring agent's equivalent of
sourcing's mock_marketplace.py.

Provides a deterministic skill rubric per role (used by the Score node) and
a small pool of interviewers/calendar slots (used by the schedule_interview
tool).
"""

import hashlib

# Skills the Score node checks the resume against, per role.
ROLE_SKILL_BANK: dict[str, list[str]] = {
    "Backend Engineer": ["python", "distributed systems", "postgres", "api design", "kubernetes"],
    "AI Engineer": ["llm", "pytorch", "rag", "prompt engineering", "vector database"],
    "Frontend Engineer": ["react", "typescript", "css", "accessibility", "performance"],
    "Data Engineer": ["sql", "airflow", "spark", "etl", "data modeling"],
}

DEFAULT_SKILL_BANK = ["communication", "problem solving", "ownership", "collaboration", "adaptability"]

MOCK_INTERVIEWERS = ["Priya Sharma", "Alex Kim", "Jordan Lee", "Sam Okafor"]

MOCK_CALENDAR_SLOTS = [
    "Tue 10:00 AM PT",
    "Tue 2:00 PM PT",
    "Wed 11:00 AM PT",
    "Thu 9:00 AM PT",
    "Fri 1:00 PM PT",
]


def skill_bank_for_role(role: str) -> list[str]:
    return ROLE_SKILL_BANK.get(role, DEFAULT_SKILL_BANK)


def extract_matched_skills(resume_text: str, role: str) -> list[str]:
    """Deterministic keyword match — resume text vs. the role's skill bank.

    This backs the Score node when no LLM is configured (or as ground
    truth to sanity-check the LLM's extracted skills). Case-insensitive
    substring match, same spirit as sourcing's case-insensitive item match.
    """
    text = resume_text.lower()
    return [skill for skill in skill_bank_for_role(role) if skill.lower() in text]


def schedule_slot(candidate_name: str, role: str) -> dict:
    """Deterministically pick an interviewer + slot for a candidate.

    Deterministic (hash-based) rather than random so the same candidate +
    role always gets the same mock slot — useful for reproducible demos
    and eval scenarios.
    """
    h = int(hashlib.sha256(f"{candidate_name}:{role}".encode()).hexdigest(), 16)
    interviewer = MOCK_INTERVIEWERS[h % len(MOCK_INTERVIEWERS)]
    slot = MOCK_CALENDAR_SLOTS[h % len(MOCK_CALENDAR_SLOTS)]
    calendar_id = f"CAL-{h % 100000:05d}"
    return {
        "candidate_name": candidate_name,
        "role": role,
        "slot": slot,
        "interviewer": interviewer,
        "calendar_id": calendar_id,
    }
