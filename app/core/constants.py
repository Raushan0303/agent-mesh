from datetime import timedelta

from temporalio.common import RetryPolicy

# ── Retry policy templates (platform) ──
# Any agent's Activities can opt into one of these instead of hand-rolling RetryPolicy.

# Aggressive retry — safe for read-only or idempotent-by-nature calls.
# Used for: read-only tools (queries, lookups, ratings).
# If these fail and retry, nothing bad happens — they're pure reads.
AGGRESSIVE_RETRY_TEMPLATE = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=10),
    maximum_attempts=10,
    non_retryable_error_types=[],
)

# Strict non-retryable — for calls where blind retry risk is real.
# Used for: side-effecting tools (orders, payments).
# Retries on transient failures (network, timeout) but NOT on ToolExecutionError
# (which means the tool itself raised — an ambiguous failure that might mean
# the side effect already happened).
STRICT_NON_RETRYABLE_TEMPLATE = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
    non_retryable_error_types=["ToolExecutionError"],
)

# Legacy alias — Week 1 used this name. Kept for backward compatibility.
STRICT_RETRY_TEMPLATE = AGGRESSIVE_RETRY_TEMPLATE

DEFAULT_ACTIVITY_TIMEOUT = timedelta(seconds=120)

# ── Per-Activity timeout configuration (Week 5) ──
# Temporal has three Activity timeout types:
#   StartToClose  — max time for the Activity to run once started
#   ScheduleToStart — max time for the Activity to be picked up by a Worker
#   ScheduleToClose — total time from scheduling to completion (retries included)
#
# Picking the wrong one hides failures:
#   - If StartToClose is too long, a hung Activity occupies a Worker slot
#   - If ScheduleToStart is too long, a dead Worker's tasks sit in the queue
#   - ScheduleToClose should be StartToClose * max_attempts + backoff time

# Read-only tools (research, score) — fast, safe to retry
READ_ONLY_START_TO_CLOSE = timedelta(seconds=30)
READ_ONLY_SCHEDULE_TO_START = timedelta(seconds=10)
READ_ONLY_SCHEDULE_TO_CLOSE = timedelta(seconds=120)

# Side-effecting tools (PO, payment) — slower, careful retry
SIDE_EFFECT_START_TO_CLOSE = timedelta(seconds=45)
SIDE_EFFECT_SCHEDULE_TO_START = timedelta(seconds=10)
SIDE_EFFECT_SCHEDULE_TO_CLOSE = timedelta(seconds=180)

# Graph activities (run_graph_until_interrupt, resume_graph) — longer
GRAPH_START_TO_CLOSE = timedelta(seconds=60)
GRAPH_SCHEDULE_TO_START = timedelta(seconds=10)
GRAPH_SCHEDULE_TO_CLOSE = timedelta(seconds=180)

# Default (used by legacy Activities)
DEFAULT_SCHEDULE_TO_START = timedelta(seconds=10)
DEFAULT_SCHEDULE_TO_CLOSE = timedelta(seconds=120)

# ── Week 12-13: SSE + Webhook constants ──

# SSE channel naming: workflow:{workflow_id}:events
SSE_CHANNEL_PREFIX = "workflow"
SSE_HISTORY_SUFFIX = "events:history"  # appended to channel name for history list

# Webhook headers
WEBHOOK_SIGNATURE_HEADER = "X-AgentMesh-Signature"
WEBHOOK_EVENT_HEADER = "X-AgentMesh-Event"
WEBHOOK_WORKFLOW_ID_HEADER = "X-AgentMesh-Workflow-Id"
WEBHOOK_DELIVERY_ID_HEADER = "X-AgentMesh-Delivery-Id"
