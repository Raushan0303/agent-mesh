"""Prompt construction and response parsing for the Decide node.

Extracted from graph.py to keep file sizes under 500 LOC.
These are pure functions — no I/O, no state, no side effects.

Prompt versioning: build_decide_prompt returns a PromptVersion containing
the messages, a SHA256 hash of the rendered prompt, and a version tag.
The hash + version are stored in the graph state and checkpointed by
LangGraph, so every past run has a permanent record of which prompt was used.
"""

import json

from app.agentmesh.llm import LLMMessage
from app.agentmesh.prompt_versioning import PromptVersion, version_prompt

# Bump this when you change the decide prompt template.
# Old runs keep their version in the checkpoint. New runs get the new one.
DECIDE_PROMPT_VERSION = "v1"


def build_decide_prompt(
    item: str,
    quantity: int,
    budget: float,
    scored_suppliers: list[dict],
    past_decisions: list[dict],
) -> PromptVersion:
    """Build the LLM prompt for the Decide node.

    Returns a PromptVersion containing:
      - messages: list of LLMMessage objects (system + user)
      - hash: SHA256[:16] of the rendered messages
      - version: human-readable version tag (e.g. "v1")
    """
    system = (
        "You are a sourcing agent that selects the best supplier for a procurement request. "
        "You are given a list of scored suppliers (sorted by best combined score) and "
        "optionally past sourcing decisions from memory. "
        "Your job is to:\n"
        "1. Select the best supplier from the list\n"
        "2. Explain your reasoning, considering price, rating, lead time, and past experience\n"
        "3. If a past decision with positive feedback exists for the same item, factor that in\n\n"
        "Respond in this exact format:\n"
        "SELECTED: <supplier_name>\n"
        "RATIONALE: <your reasoning in 2-3 sentences mentioning price, rating, and lead time>"
    )

    # Build supplier table
    supplier_lines = []
    for i, s in enumerate(scored_suppliers):
        supplier_lines.append(
            f"  {i+1}. {s['name']} — price=${s['price']}/unit, "
            f"rating={s['rating']}/5, lead_time={s['lead_time_days']}d"
        )
    suppliers_text = "\n".join(supplier_lines)

    # Build past decisions context
    past_text = "No past sourcing decisions found."
    if past_decisions:
        past_lines = []
        for pd in past_decisions[:3]:
            meta = pd.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            past_lines.append(
                f"  - {pd.get('content', '')[:100]}..."
            )
        past_text = "\n".join(past_lines)

    user = (
        f"Procurement Request:\n"
        f"  Item: {item}\n"
        f"  Quantity: {quantity}\n"
        f"  Budget: ${budget}/unit\n\n"
        f"Scored Suppliers (best first):\n{suppliers_text}\n\n"
        f"Past Sourcing Decisions (from memory):\n{past_text}\n\n"
        f"Select the best supplier and explain your reasoning."
    )

    return version_prompt(
        [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ],
        version=DECIDE_PROMPT_VERSION,
    )


def parse_llm_decision(content: str, scored_suppliers: list[dict]) -> tuple[dict, str]:
    """Parse the LLM response into (selected_supplier, rationale).

    Expected format:
      SELECTED: <supplier_name>
      RATIONALE: <reasoning>

    Falls back to scored[0] if parsing fails.
    """
    best = scored_suppliers[0]
    rationale = content.strip()

    selected = None
    for line in content.split("\n"):
        if line.strip().upper().startswith("SELECTED:"):
            name = line.split(":", 1)[1].strip()
            # Match to a supplier (case-insensitive)
            for s in scored_suppliers:
                if s["name"].lower() == name.lower():
                    selected = s
                    break
            break

    if selected is None:
        # Fallback: use the top-scored supplier
        selected = best

    # Extract rationale
    for line in content.split("\n"):
        if line.strip().upper().startswith("RATIONALE:"):
            rationale = line.split(":", 1)[1].strip()
            break

    return selected, rationale
