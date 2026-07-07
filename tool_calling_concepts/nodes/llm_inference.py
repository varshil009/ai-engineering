"""LLM Inference Node — calls Groq with the system prompt and user query.

Decides whether to:
1. Call the ``execute_supabase_query`` tool (returns tool_calls),
2. Respond directly to the user (sets ``response`` and ends the graph).
"""

import json
from typing import Any

from tool_calling_concepts.config import TABLE_SCHEMAS
from tool_calling_concepts.models.schemas import AgentState
from tool_calling_concepts.models.tools import get_execute_supabase_query_tool
from tool_calling_concepts.services.groq_client import GroqClient


def _build_system_prompt() -> str:
    """Construct the system prompt with full table schema context."""
    schema_blocks: list[str] = []
    # creates a list of strings, each representing a table and its columns, to be included in the system prompt
    for table_name, columns in TABLE_SCHEMAS.items():
        col_lines = "\n".join(
            f"  - {col['column']} ({col['type']}): {col['description']}"
            for col in columns
        )
        #print(f"llm_inference._build_system_prompt: table {table_name} columns:\n{col_lines}")
        schema_blocks.append(f"### Table: {table_name}\n{col_lines}")

    schemas_text = "\n\n".join(schema_blocks)
    #print(f"llm_inference._build_system_prompt.schemas_text:\n{schemas_text}")
    return (
        "You are an agent, You have access to a Supabase PostgreSQL, you have to return the table that was returned from tool call. "
        "database containing transformer and prediction data.\n\n"
        "## Database Schema\n\n"
        f"{schemas_text}\n\n"
        "## Rules\n"
        "1. You can generate SQL SELECT queries to answer questions about the data.\n"
        "2. Always use the `execute_supabase_query` tool to run queries — do NOT fabricate results.\n"
        "3. Only generate SELECT queries. Never INSERT, UPDATE, DELETE, DROP, ALTER, etc.\n"
        "4. Use proper PostgreSQL syntax. The `Date` column in BOLNEY is `timestamp with time zone`.\n"
        "5. When comparing timestamps, use proper timestamp comparisons (e.g. `>= NOW() - INTERVAL '7 days'`).\n"
        "6. If the user asks a question that doesn't require a database query, answer directly.\n"
        "7. Keep responses concise but informative. Include the SQL you ran when relevant.\n"
        "8. The SGT tables (SGT1-SGT4) contain prediction data — use `predicted_for_utc` for the prediction target time.\n"
        "9. BOLNEY contains actual active power readings.\n"
        "10. You have to show all the data returned as markdown tables in your response. If the data is too large, show the first 10 rows and indicate that there are more rows.\n"
        "⚠️⚠️⚠️ MOST IMPORTANT ⚠️⚠️⚠️\n"
        "11. If the user asks for a table that does not exist, inform them that the table does not exist and do not attempt to call the tool."
        "⚠️⚠️⚠️ ANOTHER MOST IMPORTANT ⚠️⚠️⚠️\n"
        "GENERATE A MARKDOWN TABLE FROM THE DATA RETURNED FROM THE TOOL CALL. WITHOUT ADDING DATA FROM YOUR SIDE"
        
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

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": query},
    ]

    try:
        result = await client.chat_completion(
            messages=messages,
            tools=[get_execute_supabase_query_tool()],
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