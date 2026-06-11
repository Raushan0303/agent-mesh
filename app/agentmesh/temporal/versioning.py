"""Workflow Versioning helper — thin wrapper around Temporal's versioning API.

This documents the AgentMesh convention for versioning Workflow code changes.
The mechanism is Temporal's; the convention for how AgentMesh Workflows use
it consistently is ours.

Convention:
- Every Workflow code change gets a unique change_id string.
- Use is_patched() for behavioral changes (new branches).
- Use deprecate_patch() when all in-flight workflows have completed
  and the old code path can be safely removed.

The Python Temporal SDK uses workflow.patched() which returns:
- True for new workflows (the patch is "seen" for the first time)
- True for replaying workflows that already have the patch marker
- False for replaying workflows that don't have the patch marker (old in-flight)
"""

from temporalio import workflow


def is_patched(change_id: str) -> bool:
    """Wrapper around Temporal's workflow.patched().

    Usage in a Workflow:
        if is_patched("add-score-step"):
            # New code path — only newly started Workflows execute this
            await workflow.execute_activity(score_activity, ...)
        # Old code path — in-flight Workflows skip the new step

    Returns True for new workflows and workflows that have already seen
    this patch. Returns False for old in-flight workflows that haven't
    seen the patch (they continue on the old code path).
    """
    return workflow.patched(change_id)


def deprecate_patch(change_id: str) -> None:
    """Wrapper around Temporal's workflow.deprecate_patch().

    Call this when all in-flight workflows with the old code path have
    completed and you want to remove the old branch. This tells Temporal
    that the patch is no longer needed and the new behavior is the only
    behavior.
    """
    workflow.deprecate_patch(change_id)
