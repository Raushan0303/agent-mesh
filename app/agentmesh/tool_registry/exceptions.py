class ToolRegistryError(Exception):
    """Base exception for all tool registry errors."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a tool name is not registered."""


class SchemaValidationError(ToolRegistryError):
    """Raised when tool input or output fails schema validation."""


class ToolTimeoutError(ToolRegistryError):
    """Raised when a tool exceeds its configured timeout."""


class ToolExecutionError(ToolRegistryError):
    """Raised when a tool raises an exception during execution."""


class EgressDeniedError(ToolRegistryError):
    """Raised when an outbound request is blocked by the egress filter."""


class ToolApprovalRequiredError(ToolRegistryError):
    """Raised when a tool with authorization_mode=APPROVAL_REQUIRED is called
    without prior approval. The workflow should catch this, pause for human
    approval, and retry after approval is received."""

