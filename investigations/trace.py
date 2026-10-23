"""Safe investigation execution events; arguments and document text are omitted."""

from investigations.schemas import ExecutionEvent


def event(*, action_id, tool_name, status, duration_ms, source_references=(), error_category=None):
    return ExecutionEvent(
        action_id=action_id,
        tool_name=tool_name,
        status=status,
        duration_ms=round(max(0.0, duration_ms), 3),
        source_references=list(source_references),
        error_category=error_category,
    )
