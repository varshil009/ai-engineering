"""Tool Executor Node — executes SQL queries via Supabase and returns results."""

import json
from typing import Any

from tool_calling_concepts.models.schemas import AgentState
from tool_calling_concepts.services.supabase_client import run_sql_query
from tool_calling_concepts.utils.sql_validator import validate_sql_query


async def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: executes each pending tool call against Supabase.

    For each tool call with name ``execute_supabase_query``:
    1. Extract the ``sql_query`` argument.
    2. Validate it's a safe SELECT query.
    3. Execute it via the Supabase client.
    4. Format the result as a tool message.

    Args:
        state: The current agent state with pending ``tool_calls``.

    Returns:
        Updated state with ``tool_results`` populated and error handling.
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"tool_results": [], "error": "No tool calls to execute."}

    tool_results: list[dict[str, Any]] = []
    last_error: str | None = None

    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        tool_name = function_info.get("name", "")
        tool_call_id = tool_call.get("id", "")

        if tool_name != "execute_supabase_query":
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
            })
            continue

        # Parse arguments
        try:
            arguments: dict[str, Any] = json.loads(function_info.get("arguments", "{}"))
            sql_query_raw: str = arguments.get("sql_query", "")
        except (json.JSONDecodeError, KeyError) as exc:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"Failed to parse tool arguments: {exc}"}),
            })
            last_error = str(exc)
            continue

        # Validate SQL
        try:
            sql_query = validate_sql_query(sql_query_raw)
        except ValueError as exc:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"SQL validation failed: {exc}"}),
                "sql_query": sql_query_raw,
            })
            last_error = str(exc)
            continue

        # Execute SQL
        try:
            rows = await run_sql_query(sql_query)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(rows, indent=2, default=str),
                "sql_query": sql_query,
                "row_count": len(rows),
            })
        except Exception as exc:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"SQL execution failed: {exc}"}),
                "sql_query": sql_query,
            })
            last_error = str(exc)

    return {
        "tool_results": tool_results,
        "error": last_error,
    }