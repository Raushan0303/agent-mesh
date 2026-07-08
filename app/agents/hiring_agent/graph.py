"""LangGraph StateGraph for the hiring agent.

Same engine as the sourcing agent (Temporal durability + LangGraph nodes +
interrupt() for human checkpoints), applied to a harder, higher-stakes
workflow: screening, interviewing, and making an offer decision on a
candidate — with a human recruiter gating the final call.

Topology (matches the architecture-page diagram exactly):

    START -> Screen Resume -> Score Rubric --[score < threshold]--> Reject -> END
                                    |
                                    v (score >= threshold)
                              Schedule Interview -> Interview (follow-up loop, <=3)
                                    |
                                    v
                              Human Review (interrupt) --[rejected, retries left]--> Screen Resume (retry w/ feedback)
                                    |                     --[rejected, out of retries]--> END
                                    v (approved)
                              Offer Decision -> END (terminal; send_offer runs as a
                                                 separate Temporal Activity, same
                                                 pattern as sourcing's create_po/
                                                 initiate_payment)

Node types (deliberately separated, same philosophy as sourcing_agent/graph.py):
  Screen Resume   — agentic (LLM extracts skills/signal from the resume)
  Score Rubric    — deterministic function (NO LLM, pure rubric scoring)
  Schedule        — agentic/tool (single MCP tool call through the registry)
  Interview       — agentic (LLM-driven, bounded follow-up loop)
  Human Review    — human checkpoint (interrupt() pauses until a recruiter decides)
  Offer Decision  — agentic (LLM proposes the final offer amount)
"""

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from app.agentmesh.tool_registry import registry
from app.agents.hiring_agent.mock_ats import extract_matched_skills, skill_bank_for_role
from app.agents.hiring_agent.state import AgentState

logger = logging.getLogger("agentmesh.hiring_agent.graph")

SCORE_THRESHOLD = 55.0
MAX_FOLLOWUPS = 3
MAX_REJECTIONS = 3


# ── Node 1: Screen Resume (agentic) ──


def _build_screen_prompt(role: str, resume_text: str, feedback: str) -> list:
    from app.agentmesh.llm import LLMMessage

    skills = skill_bank_for_role(role)
    system = (
        "You are a technical recruiter screening a resume for a specific role. "
        "You are given the role's key skills and the candidate's resume text. "
        "Extract which of the role's key skills the candidate demonstrates.\n\n"
        "Respond in this exact format:\n"
        "MATCHED_SKILLS: <comma-separated list of matched skills from the list below>\n"
        "SUMMARY: <2-3 sentence summary of the candidate's fit>"
    )
    feedback_text = f"\n\nPrevious recruiter feedback to consider: {feedback}" if feedback else ""
    user = (
        f"Role: {role}\n"
        f"Key skills for this role: {', '.join(skills)}\n\n"
        f"Resume:\n{resume_text}"
        f"{feedback_text}\n\n"
        f"Extract matched skills and summarize fit."
    )
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]


def _parse_screen_response(content: str, role: str, resume_text: str) -> tuple[list[str], str]:
    """Parse MATCHED_SKILLS / SUMMARY from the LLM response.

    Falls back to deterministic keyword matching (mock_ats.extract_matched_skills)
    if the LLM response doesn't follow the expected format — same
    resilience pattern as sourcing's _parse_llm_decision.
    """
    matched: list[str] | None = None
    summary = content.strip()

    for line in content.split("\n"):
        if line.strip().upper().startswith("MATCHED_SKILLS:"):
            raw = line.split(":", 1)[1].strip()
            skills = skill_bank_for_role(role)
            matched = [s for s in skills if s.lower() in raw.lower()]
        if line.strip().upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    if matched is None:
        matched = extract_matched_skills(resume_text, role)

    return matched, summary


async def screen_resume_node(state: AgentState) -> dict:
    """Screen Resume node: LLM extracts skill signal from the resume.

    Runs once per pass through the graph. If the human rejects the
    candidate at Human Review with feedback and retries remain, the graph
    loops back here and the feedback is folded into the prompt, so the
    re-screen genuinely considers what the recruiter said.
    """
    brief = state["brief"]
    feedback = state.get("screening_feedback", "")

    messages = _build_screen_prompt(brief.role, brief.resume_text, feedback)

    try:
        from app.agentmesh.llm import get_llm_client
        client = get_llm_client()
        response = await client.complete(messages, temperature=0.0, max_tokens=400)
        matched, summary = _parse_screen_response(response.content, brief.role, brief.resume_text)
    except Exception as e:
        logger.warning("SCREEN_LLM_FAILED error=%s — falling back to keyword match", e)
        matched = extract_matched_skills(brief.resume_text, brief.role)
        summary = f"Keyword match found {len(matched)} of the role's key skills in the resume."

    logger.info(
        "SCREEN_COMPLETED candidate=%s role=%s matched_skills=%d",
        brief.candidate_name,
        brief.role,
        len(matched),
    )
    return {"skills_matched": matched, "screening_notes": summary}


# ── Node 2: Score Rubric (deterministic function — NO LLM) ──


def score_node(state: AgentState) -> dict:
    """Score Rubric node: deterministic scoring, same philosophy as
    sourcing's score_node — no LLM call, no randomness, no I/O.

    score = 60 * (matched_skills / total_skills) + min(years_experience, 10) * 4
    """
    brief = state["brief"]
    matched = state.get("skills_matched", [])
    total_skills = max(len(skill_bank_for_role(brief.role)), 1)

    skill_component = 60.0 * (len(matched) / total_skills)
    experience_component = min(brief.years_experience, 10.0) * 4.0
    score = round(skill_component + experience_component, 1)

    logger.info(
        "SCORE_COMPLETED candidate=%s score=%.1f matched=%d/%d experience=%.1fy",
        brief.candidate_name,
        score,
        len(matched),
        total_skills,
        brief.years_experience,
    )
    return {"screening_score": score}


def reject_node(state: AgentState) -> dict:
    """Reject node: score didn't clear the bar — the interview never happens."""
    logger.info(
        "REJECT_BY_SCORE candidate=%s score=%.1f threshold=%.1f",
        state["brief"].candidate_name,
        state.get("screening_score", 0.0),
        SCORE_THRESHOLD,
    )
    return {"final_status": "rejected_by_score"}


# ── Node 3: Schedule Interview (tool call via the MCP Tool Registry) ──


async def schedule_interview_node(state: AgentState) -> dict:
    brief = state["brief"]
    result = await registry.call(
        "schedule_interview",
        {"candidate_name": brief.candidate_name, "role": brief.role},
    )
    logger.info(
        "SCHEDULE_COMPLETED candidate=%s interviewer=%s slot=%s",
        brief.candidate_name,
        result.get("interviewer"),
        result.get("slot"),
    )
    return {"interview_slot": result}


# ── Node 4: Interview (agentic, bounded follow-up loop) ──


def _build_interview_prompt(brief, matched_skills: list[str], followup_count: int, transcript: list[dict]) -> list:
    from app.agentmesh.llm import LLMMessage

    system = (
        "You are conducting a technical interview. You are given the candidate's "
        "matched skills from resume screening and the interview so far. Decide "
        "whether a follow-up question is needed to probe deeper, and give a "
        "score delta for this turn.\n\n"
        "Respond in this exact format:\n"
        "FOLLOWUP: <yes or no>\n"
        "SCORE_DELTA: <number from -10 to 15>\n"
        "NOTE: <one sentence on what was probed this turn>"
    )
    history = "\n".join(f"  Turn {t['turn']}: {t['notes']}" for t in transcript) or "  (no turns yet)"
    user = (
        f"Candidate: {brief.candidate_name}\nRole: {brief.role}\n"
        f"Matched skills: {', '.join(matched_skills) or 'none'}\n"
        f"Turns so far ({followup_count}):\n{history}\n\n"
        f"Continue the interview."
    )
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]


def _parse_interview_response(content: str, followup_count: int, matched_skills: list[str]) -> tuple[bool, float, str]:
    """Parse FOLLOWUP / SCORE_DELTA / NOTE. Falls back to a deterministic
    rule when the LLM response doesn't match: probe once more if fewer than
    half the role's skills matched, otherwise wrap up."""
    needs_followup = None
    score_delta = None
    note = content.strip()[:200]

    for line in content.split("\n"):
        upper = line.strip().upper()
        if upper.startswith("FOLLOWUP:"):
            needs_followup = "yes" in line.split(":", 1)[1].strip().lower()
        elif upper.startswith("SCORE_DELTA:"):
            try:
                score_delta = float(line.split(":", 1)[1].strip())
            except ValueError:
                score_delta = None
        elif upper.startswith("NOTE:"):
            note = line.split(":", 1)[1].strip()

    if needs_followup is None:
        needs_followup = followup_count == 0 and len(matched_skills) < 3
    if score_delta is None:
        score_delta = 8.0 if matched_skills else 4.0

    return needs_followup, score_delta, note


async def interview_node(state: AgentState) -> dict:
    """Interview node: one turn of the interview per invocation.

    The conditional edge below re-enters this node (a genuine self-loop in
    the graph, not a Python for-loop) while FOLLOWUP=yes and the bounded
    limit (MAX_FOLLOWUPS) hasn't been hit — this is the "follow-up loop <=3"
    branch on the architecture diagram.
    """
    brief = state["brief"]
    followup_count = state.get("followup_count", 0)
    matched = state.get("skills_matched", [])
    transcript = state.get("interview_transcript", [])

    messages = _build_interview_prompt(brief, matched, followup_count, transcript)

    try:
        from app.agentmesh.llm import get_llm_client
        client = get_llm_client()
        response = await client.complete(messages, temperature=0.3, max_tokens=300)
        needs_followup, score_delta, note = _parse_interview_response(response.content, followup_count, matched)
    except Exception as e:
        logger.warning("INTERVIEW_LLM_FAILED error=%s — using deterministic turn", e)
        needs_followup = followup_count == 0 and len(matched) < 3
        score_delta = 8.0 if matched else 4.0
        note = "Deterministic fallback turn (no LLM provider configured)."

    new_transcript = transcript + [{"turn": followup_count + 1, "notes": note}]
    new_score = state.get("interview_score", 50.0) + score_delta
    new_followup_count = followup_count + 1
    done = (not needs_followup) or new_followup_count >= MAX_FOLLOWUPS

    logger.info(
        "INTERVIEW_TURN candidate=%s turn=%d score=%.1f done=%s",
        brief.candidate_name,
        new_followup_count,
        new_score,
        done,
    )
    return {
        "interview_transcript": new_transcript,
        "followup_count": new_followup_count,
        "interview_score": round(new_score, 1),
        "interview_done": done,
    }


# ── Node 5: Human Review (human checkpoint via interrupt()) ──


def human_review_node(state: AgentState) -> dict:
    """Human Review node: structural checkpoint via LangGraph's interrupt().

    Same rationale as sourcing's approve_node — the graph genuinely
    suspends here. A recruiter must call Command(resume=review_data) via
    the platform's agent-agnostic POST /workflows/{id}/approve route.
    """
    brief = state["brief"]

    review_request = {
        "candidate_name": brief.candidate_name,
        "role": brief.role,
        "screening_score": state.get("screening_score", 0.0),
        "interview_score": state.get("interview_score", 0.0),
        "skills_matched": state.get("skills_matched", []),
        "interview_slot": state.get("interview_slot", {}),
    }

    logger.info(
        "HUMAN_REVIEW_INTERRUPT_FIRED candidate=%s screening_score=%.1f interview_score=%.1f",
        brief.candidate_name,
        review_request["screening_score"],
        review_request["interview_score"],
    )

    review = interrupt(review_request)
    approved = review.get("approved", False)
    comment = review.get("comment", "")

    logger.info("HUMAN_REVIEW_RESUMED approved=%s comment=%s", approved, comment)

    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_comment": comment,
        "rejection_count": state.get("rejection_count", 0) + (0 if approved else 1),
    }


# ── Node 6: Offer Decision (agentic — finalizes the offer amount) ──


def _build_offer_prompt(brief, interview_score: float):
    from app.agentmesh.llm import LLMMessage

    system = (
        "You are finalizing a job offer. Given the candidate's target salary, "
        "years of experience, and interview score, propose a fair offer amount.\n\n"
        "Respond in this exact format:\n"
        "OFFER_AMOUNT: <number>\n"
        "RATIONALE: <1-2 sentences>"
    )
    user = (
        f"Candidate: {brief.candidate_name}\nRole: {brief.role}\n"
        f"Target salary: ${brief.target_salary:,.0f}\n"
        f"Years of experience: {brief.years_experience}\n"
        f"Interview score: {interview_score:.1f}/100\n\n"
        f"Propose the offer amount."
    )
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]


def _parse_offer_response(content: str, brief, interview_score: float) -> tuple[float, str]:
    amount = None
    rationale = content.strip()

    for line in content.split("\n"):
        upper = line.strip().upper()
        if upper.startswith("OFFER_AMOUNT:"):
            raw = line.split(":", 1)[1].strip().replace("$", "").replace(",", "")
            try:
                amount = float(raw)
            except ValueError:
                amount = None
        if upper.startswith("RATIONALE:"):
            rationale = line.split(":", 1)[1].strip()

    if amount is None:
        multiplier = 1.0 if interview_score >= 80 else 0.93
        amount = round(brief.target_salary * multiplier, -2)
        rationale = (
            f"Deterministic fallback: interview score {interview_score:.1f} -> "
            f"{multiplier:.0%} of target salary ${brief.target_salary:,.0f}."
        )

    return amount, rationale


async def offer_decision_node(state: AgentState) -> dict:
    """Offer Decision node: proposes the final offer amount.

    Terminal node of the graph itself — the actual `send_offer` side
    effect is a separate Temporal Activity (see workflow.py), same
    separation of concerns as sourcing's create_po/initiate_payment
    running after the graph resumes and completes.
    """
    brief = state["brief"]
    interview_score = state.get("interview_score", 50.0)

    messages = _build_offer_prompt(brief, interview_score)
    try:
        from app.agentmesh.llm import get_llm_client
        client = get_llm_client()
        response = await client.complete(messages, temperature=0.0, max_tokens=300)
        amount, rationale = _parse_offer_response(response.content, brief, interview_score)
    except Exception as e:
        logger.warning("OFFER_LLM_FAILED error=%s — falling back to deterministic offer", e)
        multiplier = 1.0 if interview_score >= 80 else 0.93
        amount = round(brief.target_salary * multiplier, -2)
        rationale = (
            f"Deterministic offer: interview score {interview_score:.1f} -> "
            f"{multiplier:.0%} of target salary ${brief.target_salary:,.0f}."
        )

    logger.info(
        "OFFER_DECISION_COMPLETED candidate=%s amount=%.0f",
        brief.candidate_name,
        amount,
    )
    return {"offer_amount": amount, "decision_reason": rationale, "final_status": "awaiting_offer"}


# ── Conditional edges ──


def _after_score(state: AgentState) -> str:
    if state.get("screening_score", 0.0) < SCORE_THRESHOLD:
        return "reject"
    return "schedule"


def _after_interview(state: AgentState) -> str:
    if state.get("interview_done", False):
        return "human_review"
    return "interview"


def _after_human_review(state: AgentState) -> str:
    if state.get("approval_status") == "approved":
        return "offer_decision"

    rejection_count = state.get("rejection_count", 0)
    if rejection_count >= MAX_REJECTIONS:
        logger.info("HUMAN_REVIEW_REJECTED_MAX_RETRIES rejections=%d — giving up", rejection_count)
        return END

    logger.info(
        "HUMAN_REVIEW_REJECTED_RETRY rejection_count=%d comment=%s — re-screening",
        rejection_count,
        state.get("approval_comment", ""),
    )
    return "screen_resume"


# ── Graph builder ──


def build_hiring_graph(checkpointer=None):
    """Build the full hiring-agent StateGraph.

    Args:
        checkpointer: A LangGraph checkpointer (AsyncPostgresSaver) — required
            for interrupt() to work, same as build_sourcing_graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("screen_resume", screen_resume_node)
    graph.add_node("score", score_node)
    graph.add_node("reject", reject_node)
    graph.add_node("schedule", schedule_interview_node)
    graph.add_node("interview", interview_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("offer_decision", offer_decision_node)

    graph.add_edge(START, "screen_resume")
    graph.add_edge("screen_resume", "score")

    graph.add_conditional_edges("score", _after_score, {"reject": "reject", "schedule": "schedule"})
    graph.add_edge("reject", END)

    graph.add_edge("schedule", "interview")
    graph.add_conditional_edges("interview", _after_interview, {"interview": "interview", "human_review": "human_review"})

    graph.add_conditional_edges(
        "human_review",
        _after_human_review,
        {"offer_decision": "offer_decision", "screen_resume": "screen_resume", END: END},
    )

    graph.add_edge("offer_decision", END)

    return graph.compile(checkpointer=checkpointer)
