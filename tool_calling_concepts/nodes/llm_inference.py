"""LLM Inference Node — calls Groq with the system prompt and user query.

Decides whether to:
1. Call the ``execute_supabase_query`` tool (returns tool_calls),
2. Call the ``execute_python_analysis`` tool to analyse large results,
3. Respond directly to the user (sets ``response`` and ends the graph).
"""

import json
from typing import Any

from tool_calling_concepts.config import TABLE_SCHEMAS
from tool_calling_concepts.models.schemas import AgentState
from tool_calling_concepts.models.tools import (
    get_execute_supabase_query_tool,
    get_terminal_tool,
)
from tool_calling_concepts.services.groq_client import GroqClient


def _build_system_prompt() -> str:
    """Construct the system prompt with full table schema context."""
    schema_blocks: list[str] = []
    for table_name, columns in TABLE_SCHEMAS.items():
        col_lines = "\n".join(
            f"  - {col['column']} ({col['type']}): {col['description']}"
            for col in columns
        )
        schema_blocks.append(f"### Table: {table_name}\n{col_lines}")

    schemas_text = "\n\n".join(schema_blocks)
    return (
        "You are an agent with access to a Supabase PostgreSQL database containing "
        "transformer and prediction data. You can also run Python code to analyse data.\n\n"
        "## Database Schema\n\n"
        f"{schemas_text}\n\n"
        "## Available Tools\n"
        "1. **execute_supabase_query** — Run SQL SELECT queries on the database.\n"
        "2. **execute_python_analysis** — Run Python code to analyse or summarise data.\n\n"
        "## Workflow\n"
        "1. First, use `execute_supabase_query` to fetch data from the database.\n"
        "2. If the SQL results are too large (more than ~6000 words / 50+ rows), "
        "use `execute_python_analysis` to analyse/summarise the data instead of returning it raw.\n"
        "3. You can call `execute_python_analysis` multiple times to refine your analysis.\n"
        "4. When you have enough information, respond directly with a concise answer.\n\n"
        "## Rules\n"
        "1. Always use `execute_supabase_query` to run queries — do NOT fabricate results.\n"
        "2. Only generate SELECT queries. Never INSERT, UPDATE, DELETE, DROP, ALTER, etc.\n"
        "3. Use proper PostgreSQL syntax. The `Date` column in BOLNEY is `timestamp with time zone`.\n"
        "4. When comparing timestamps, use proper timestamp comparisons (e.g. `>= NOW() - INTERVAL '7 days'`).\n"
        "5. If the user asks a question that doesn't require a database query, answer directly.\n"
        "6. Keep responses concise but informative. Include the SQL you ran when relevant.\n"
        "7. The SGT tables (SGT1-SGT4) contain prediction data — use `predicted_for_utc` for the prediction target time.\n"
        "8. BOLNEY contains actual active power readings.\n"
        "9. Show data as markdown tables. If data is too large, show first 10 rows and indicate more exist.\n\n"
        "## Python Analysis Rules\n"
        "1. Use `execute_python_analysis` when SQL returned too many rows to display directly.\n"
        "2. Write Python code to: count rows, get top 10, compute basic stats (min, max, mean, etc.).\n"
        "3. Do NOT use the terminal tool for: installing packages (`pip`, `uv add`), "
        "navigating directories (`cd`), exploring the filesystem (`ls`, `dir`, `os.listdir`), "
        "or importing unexpected modules.\n"
        "4. The terminal runs in the project's uv environment. Standard library, json, and pandas are available.\n"
        "5. If a single response would exceed ~6000 words, use the terminal to analyse/summarise instead.\n"
        "6. You can call the terminal tool repeatedly until you have the answer you need.\n\n"
        "⚠️⚠️⚠️ MOST IMPORTANT ⚠️⚠️⚠️\n"
        "If the user asks for a table that does not exist, inform them and do not attempt to call the tool.\n"
        "⚠️⚠️⚠️ ANOTHER MOST IMPORTANT ⚠️⚠️⚠️\n"
        "GENERATE A MARKDOWN TABLE FROM THE DATA RETURNED FROM THE TOOL CALL. WITHOUT ADDING DATA FROM YOUR SIDE."
    )


async def llm_inference_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: sends the query to Groq and processes the response.

    If Groq calls a tool, the tool_calls are stored in state for the
    next node. If Groq responds directly, the response text is stored
    and the graph can end.
    """
    client = GroqClient()
    query = state.get("query", "").strip()

    if not query:
        return {"response": "No query provided.", "error": None}

    # Build messages from accumulated conversation history
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": query},
    ]

    # Append any accumulated conversation (tool results, previous assistant messages)
    existing_messages = state.get("messages", [])
    for msg in existing_messages:
        if msg.get("role") in ("assistant", "tool"):
            messages.append(msg)

    try:
        result = await client.chat_completion(
            messages=messages,
            tools=[get_execute_supabase_query_tool(), get_terminal_tool()],
            tool_choice="auto",
        )
    except Exception as exc:
        return {
            "response": None,
            "error": f"LLM inference failed: {exc}",
        }

    # Extract the assistant message
    choice = result["choices"][0]
    message = choice["message"]

    tool_calls = message.get("tool_calls", [])
    content = message.get("content")

    if tool_calls:
        # Store tool calls for the executor node
        return {
            "messages": messages + [message],
            "tool_calls": tool_calls,
            "response": None,
            "error": None,
        }

    # No tool calls — LLM responded directly
    return {
        "messages": messages + [message],
        "tool_calls": [],
        "response": content or "I have no specific answer to that question.",
        "error": None,
    }
