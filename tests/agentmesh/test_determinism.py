import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r"datetime\.now",
    r"datetime\.utcnow",
    r"random\.",
    r"requests\.",
    r"httpx\.",
    r"asyncpg\.",
    r"psycopg",
    r"uuid\.uuid4",
    r"time\.sleep",
]


def test_no_non_deterministic_code_in_workflow():
    """Scan workflow.py for forbidden non-deterministic patterns."""
    workflow_file = Path("app/agents/sourcing_agent/workflow.py")
    content = workflow_file.read_text()
    for pattern in FORBIDDEN_PATTERNS:
        assert not re.search(pattern, content), (
            f"Forbidden pattern in workflow.py: {pattern}"
        )
