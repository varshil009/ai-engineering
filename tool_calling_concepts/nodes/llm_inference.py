"""LLM Inference Node — calls Groq with the system prompt and user query.

Decides whether to:
1. Call the ``execute_supabase_query`` tool (returns tool_calls),
2. Call the ``execute_python_analysis`` tool to analyse large results,
3. Respond directly to the user (sets ``response`` and ends the graph).
"""

from typing import Any

from tool_calling_concepts.config import TABLE_SCHEMAS
from tool_calling_concepts.models.schemas import AgentState
from tool_calling_concepts.models.tools import (
    get_execute_supabase_query_tool,
    get_terminal_tool,
)
from tool_calling_concepts.services.groq_client import GroqClient


def _build_system_prompt() -> str:
    """Construct a concise system prompt."""
    # Build compact schema reference
    schema_parts: list[str] = []
    for table_name, columns in TABLE_SCHEMAS.items():
        cols = ", ".join(f"{c['column']} ({c['type']})" for c in columns)
        schema_parts.append(f"{table_name}: {cols}")
    schema_text = "\n".join(schema_parts)

    return (
        "You are a data agent with access to a Supabase PostgreSQL database.\n\n"
        "## Tables\n"
        f"{schema_text}\n\n"
        "## Instructions\n"
        "- Use `execute_supabase_query` to run SELECT queries. Never fabricate data.\n"
        "- Only SELECT queries allowed (no INSERT/UPDATE/DELETE/DROP/ALTER).\n"
        "- If results are large (>50 rows), use `execute_python_analysis` to summarise.\n"
        "- Show results as markdown tables. If too many rows, show first 10 and note more exist.\n"
        "- Keep answers concise but include the SQL you ran.\n"
        "- If a table doesn't exist, inform the user — don't call the tool."
    )


async def llm_inference_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: sends the query to Groq and processes the response.

    If Groq calls a tool, the tool_calls are stored in state for the
    next node. If Groq responds directly, the response text is stored
    and the graph can end.
    """
    import sys
    query = state.get("query", "").strip()
    print(f"[DEBUG llm_inference_node] Entered with query: {query[:50]}...", flush=True)
    sys.stdout.flush()

    if not query:
        print("[DEBUG llm_inference_node] No query provided, returning early", flush=True)
        return {"response": "No query provided.", "error": None}

    client = GroqClient()

    # Build messages from accumulated conversation history
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": query},
    ]

    # Append any accumulated conversation (tool results, previous assistant messages)
    existing_messages = state.get("messages", [])
    for msg in existing_messages:
        if msg.get("role") in ("assistant", "tool"):
            # Strip unsupported fields that Groq's API rejects
            # Standard OpenAI fields for assistant: role, content, tool_calls, name, refusal
            # Standard OpenAI fields for tool: role, tool_call_id, content
            # Groq rejects: annotations, executed_tools, function_call (when null), and other custom fields
            _unsupported = {"annotations", "executed_tools", "sql_query", "row_count", "terminal_output"}
            clean_msg = {k: v for k, v in msg.items() if k not in _unsupported}
            # Also remove function_call if it's None (Groq rejects null function_call)
            if clean_msg.get("function_call") is None:
                clean_msg.pop("function_call", None)
            messages.append(clean_msg)

    # Also append tool_results from state as tool messages (for loop-back)
    tool_results = state.get("tool_results", [])
    for tr in tool_results:
        if tr.get("role") == "tool" and tr not in messages:
            # Strip unsupported fields from tool messages too
            _unsupported = {"annotations", "executed_tools", "sql_query", "row_count", "terminal_output"}
            clean_tr = {k: v for k, v in tr.items() if k not in _unsupported}
            messages.append(clean_tr)

    print(f"[DEBUG llm_inference_node] Calling Groq API with {len(messages)} messages...", flush=True)
    sys.stdout.flush()
    try:
        result = await client.chat_completion(
            messages=messages,
            tools=[get_execute_supabase_query_tool(), get_terminal_tool()],
            tool_choice="auto",
        )
    except Exception as exc:
        error_msg = f"LLM inference failed: {exc}"
        print(f"[ERROR llm_inference_node] {error_msg}", flush=True)
        sys.stderr.flush()
        return {
            "response": None,
            "error": error_msg,
            "tool_calls": [],  # Clear tool_calls to prevent infinite loop back to executor
        }
    print(f"[DEBUG llm_inference_node] Groq API returned successfully", flush=True)
    print(f"[DEBUG llm_inference_node] Result keys: {list(result.keys())}", flush=True)
    sys.stdout.flush()

    # Extract the assistant message
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError) as exc:
        error_msg = f"Failed to parse LLM response structure: {exc}. Result: {str(result)[:200]}"
        print(f"[ERROR llm_inference_node] {error_msg}", flush=True)
        sys.stderr.flush()
        return {
            "response": None,
            "error": error_msg,
        }

    tool_calls = message.get("tool_calls", [])
    content = message.get("content")

    if tool_calls:
        tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        print(f"[DEBUG llm_inference_node] Tool calls generated: {tool_names}", flush=True)
        sys.stdout.flush()
        # Store tool calls for the executor node
        return {
            "messages": messages + [message],
            "tool_calls": tool_calls,
            "response": None,
            "error": None,
        }

    # No tool calls — LLM responded directly
    response_preview = (content or "")[:100]
    print(f"[DEBUG llm_inference_node] Direct response (no tool calls): {response_preview}...", flush=True)
    sys.stdout.flush()
    return {
        "messages": messages + [message],
        "tool_calls": [],
        "response": content or "I have no specific answer to that question.",
        "error": None,
    }
