TASK_QUEUE = "hiring-task-queue"

from app.agentmesh.gateway.registry import AGENT_REGISTRY, AgentRegistration
from app.agents.hiring_agent.state import HiringBriefInput
from app.agents.hiring_agent.tools import register_hiring_tools
from app.agents.hiring_agent.workflow import HiringWorkflow

AGENT_REGISTRY["hiring_agent"] = AgentRegistration(
    workflow_class=HiringWorkflow,
    task_queue=TASK_QUEUE,
    input_model=HiringBriefInput,
)

# Register all hiring tools into the platform Tool Registry at import time —
# same shared registry the sourcing agent's tools live in.
register_hiring_tools()
