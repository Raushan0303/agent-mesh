from dataclasses import dataclass


@dataclass
class AgentRegistration:
    workflow_class: type
    task_queue: str
    input_model: type


AGENT_REGISTRY: dict[str, AgentRegistration] = {}
