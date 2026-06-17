"""Feedback loop — thumbs up/down capture and weekly drift report.

Stores per-response feedback (thumbs up/down) in Postgres and generates
a weekly drift report comparing eval scores and feedback acceptance rate
over time, flagging silent quality regressions from prompt/model changes.

Agent-agnostic: the feedback table is keyed by agent_type, not by any
agent-specific concept.
"""

import logging
import time
import uuid
from dataclasses import dataclass

import asyncpg

from app.core.config import settings

logger = logging.getLogger("agentmesh.evals.drift_report")


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    id: str
    agent_type: str
    workflow_id: str
    thumbs_up: bool
    comment: str
    created_at: float


class FeedbackStore:
    """Stores and queries feedback entries — agent-agnostic."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def setup(self) -> None:
        """Create the feedback table if it doesn't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_feedback (
                    id          TEXT PRIMARY KEY,
                    agent_type  TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    thumbs_up   BOOLEAN NOT NULL,
                    comment     TEXT NOT NULL DEFAULT '',
                    created_at  DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_feedback_agent_type
                ON agent_feedback(agent_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_feedback_created_at
                ON agent_feedback(created_at)
            """)
            logger.info("FEEDBACK_STORE_INITIALIZED")

    async def record(
        self,
        agent_type: str,
        workflow_id: str,
        thumbs_up: bool,
        comment: str = "",
    ) -> str:
        """Record a feedback entry. Returns the entry ID."""
        entry_id = str(uuid.uuid4())
        now = time.time()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_feedback (id, agent_type, workflow_id, thumbs_up, comment, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                entry_id,
                agent_type,
                workflow_id,
                thumbs_up,
                comment,
                now,
            )

        logger.info(
            "FEEDBACK_RECORDED agent=%s workflow=%s thumbs_up=%s",
            agent_type,
            workflow_id,
            thumbs_up,
        )
        return entry_id

    async def get_acceptance_rate(
        self, agent_type: str, since: float = 0.0
    ) -> float:
        """Get the thumbs-up acceptance rate since a timestamp."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    count(*) as total,
                    count(*) FILTER (WHERE thumbs_up) as up
                FROM agent_feedback
                WHERE agent_type = $1 AND created_at >= $2
                """,
                agent_type,
                since,
            )

        total = int(row["total"]) if row else 0
        up = int(row["up"]) if row else 0
        return up / total if total > 0 else 0.0

    async def get_feedback_count(self, agent_type: str = None) -> int:
        """Get total feedback count."""
        async with self.pool.acquire() as conn:
            if agent_type:
                return await conn.fetchval(
                    "SELECT count(*) FROM agent_feedback WHERE agent_type = $1",
                    agent_type,
                )
            return await conn.fetchval("SELECT count(*) FROM agent_feedback")

    async def clear(self, agent_type: str = None) -> int:
        """Clear feedback entries. Returns count deleted."""
        async with self.pool.acquire() as conn:
            if agent_type:
                return await conn.fetchval(
                    "DELETE FROM agent_feedback WHERE agent_type = $1",
                    agent_type,
                )
            return await conn.fetchval("DELETE FROM agent_feedback")


@dataclass
class DriftReport:
    """Weekly drift report — compares eval scores and feedback over time."""
    agent_type: str
    current_eval_score: float
    previous_eval_score: float
    current_acceptance_rate: float
    previous_acceptance_rate: float
    eval_delta: float
    acceptance_delta: float
    drift_detected: bool
    warnings: list[str]


def generate_drift_report(
    agent_type: str,
    current_eval_score: float,
    previous_eval_score: float,
    current_acceptance_rate: float,
    previous_acceptance_rate: float,
) -> DriftReport:
    """Generate a drift report comparing current vs previous metrics.

    Flags silent quality regressions: if eval score or acceptance rate
    drops by more than 5%, drift is detected.
    """
    eval_delta = current_eval_score - previous_eval_score
    acceptance_delta = current_acceptance_rate - previous_acceptance_rate

    warnings = []
    drift_detected = False

    if eval_delta < -0.05:
        warnings.append(
            f"Eval score dropped by {abs(eval_delta):.2%} "
            f"({previous_eval_score:.2%} → {current_eval_score:.2%})"
        )
        drift_detected = True

    if acceptance_delta < -0.05:
        warnings.append(
            f"Acceptance rate dropped by {abs(acceptance_delta):.2%} "
            f"({previous_acceptance_rate:.2%} → {current_acceptance_rate:.2%})"
        )
        drift_detected = True

    report = DriftReport(
        agent_type=agent_type,
        current_eval_score=current_eval_score,
        previous_eval_score=previous_eval_score,
        current_acceptance_rate=current_acceptance_rate,
        previous_acceptance_rate=previous_acceptance_rate,
        eval_delta=eval_delta,
        acceptance_delta=acceptance_delta,
        drift_detected=drift_detected,
        warnings=warnings,
    )

    if drift_detected:
        logger.warning(
            "DRIFT_DETECTED agent=%s warnings=%s",
            agent_type,
            warnings,
        )
    else:
        logger.info(
            "DRIFT_CHECK_PASSED agent=%s eval_delta=%.4f acceptance_delta=%.4f",
            agent_type,
            eval_delta,
            acceptance_delta,
        )

    return report


# ── Singleton ──

_store: FeedbackStore | None = None
_pool: asyncpg.Pool | None = None


async def get_feedback_store() -> FeedbackStore:
    """Get or create the feedback store singleton."""
    global _store, _pool
    if _store is None:
        _pool = await asyncpg.create_pool(
            settings.agent_db_dsn,
            min_size=2,
            max_size=10,
        )
        _store = FeedbackStore(_pool)
        await _store.setup()
    return _store


async def close_feedback_store() -> None:
    """Close the feedback store pool (for tests)."""
    global _store, _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        _store = None
