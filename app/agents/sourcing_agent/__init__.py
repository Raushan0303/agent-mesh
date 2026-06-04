TASK_QUEUE = "sourcing-task-queue"

from app.agentmesh.gateway.registry import AGENT_REGISTRY, AgentRegistration
from app.agents.sourcing_agent.state import SourcingBriefInput
from app.agents.sourcing_agent.tools import register_sourcing_tools
from app.agents.sourcing_agent.workflow import SourcingWorkflow

AGENT_REGISTRY["sourcing_agent"] = AgentRegistration(
    workflow_class=SourcingWorkflow,
    task_queue=TASK_QUEUE,
    input_model=SourcingBriefInput,
)

# Register all sourcing tools into the platform Tool Registry at import time
register_sourcing_tools()
