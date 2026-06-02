from pydantic import BaseModel


class WorkflowStartRequest(BaseModel):
    agent_type: str
    input: dict
    tenant_id: str | None = None


class WorkflowStartResponse(BaseModel):
    workflow_id: str
    agent_type: str
