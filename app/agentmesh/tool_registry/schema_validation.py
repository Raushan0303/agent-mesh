from pydantic import BaseModel, ValidationError

from app.agentmesh.tool_registry.exceptions import SchemaValidationError


def validate_input(input_model: type[BaseModel], args: dict) -> BaseModel:
    """Validate tool input args against the registered Pydantic model.

    Returns the validated model instance if valid.
    Raises SchemaValidationError if validation fails.
    """
    try:
        return input_model(**args)
    except ValidationError as e:
        raise SchemaValidationError(
            f"Input validation failed for {input_model.__name__}: {e.errors()}"
        ) from e


def validate_output(output_model: type[BaseModel], result: dict) -> BaseModel:
    """Validate tool output against the registered Pydantic model.

    Returns the validated model instance if valid.
    Raises SchemaValidationError if validation fails.
    """
    try:
        return output_model(**result)
    except ValidationError as e:
        raise SchemaValidationError(
            f"Output validation failed for {output_model.__name__}: {e.errors()}"
        ) from e
