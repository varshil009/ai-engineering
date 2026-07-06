"""Response Formatter Node — takes tool results and generates a natural-language answer."""

import json
from typing import Any

from tool_calling_concepts.models.schemas import AgentState
from tool_calling_concepts.services.groq_client import GroqClient


async def response_formatter_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: synthesises a final answer from tool results.

    Sends the original query, the SQL that was run, and the query results
    back to the LLM to generate a concise, natural-language response.

    Args:
        state: The current agent state with ``tool_results`` populated.

    Returns:
        Updated state with the final ``response``.
    """
    # If the LLM already responded directly (no tool calls), skip re-querying
    if state.get("response"):
        return {"response": state["response"]}

    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    messages = state.get("messages", [])

    if not tool_results:
        return {
            "response": "No data was retrieved. Unable to answer the query.",
            "error": state.get("error"),
        }

    # Build a summary of tool results for the LLM
    results_summary: list[dict[str, Any]] = []
    for tr in tool_results:
        try:
            content = json.loads(tr.get("content", "[]"))
        except (json.JSONDecodeError, TypeError):
            content = tr.get("content", "")

        sql_query = tr.get("sql_query", "")
        row_count = tr.get("row_count", 0)

        results_summary.append({
            "sql_query": sql_query,
            "row_count": row_count,
            "data": content if isinstance(content, list) else [content],
        })

    # If there was an SQL error, return it directly
    if state.get("error"):
        return {
            "response": (
                f"I encountered an error while querying the database: {state['error']}\n\n"
                f"The SQL query that was attempted:\n```sql\n{tool_results[0].get('sql_query', 'N/A')}\n```"
            ),
            "error": state["error"],
        }

    system_prompt = (
        "You are a power-grid data analyst assistant. Your role is to take SQL query results "
        "and explain them to the user in clear, natural language. "
        "Provide relevant numbers, trends, and insights. Be concise but thorough. "
        "If the data contains timestamps, express them in a human-readable format. "
        "Do NOT fabricate data — only report what the query returned."
    )

    llm_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Include prior conversation context (user + assistant messages)
    # to help the LLM understand the full picture
    user_message_content = (
        f"Original user query: {query}\n\n"
        f"SQL query results:\n```json\n{json.dumps(results_summary, indent=2, default=str)}\n```\n\n"
        "Please provide a clear natural-language answer based on these results."
    )
    llm_messages.append({"role": "user", "content": user_message_content})

    try:
        client = GroqClient()
        result = await client.chat_completion(
            messages=llm_messages,
            tools=None,  # No tools needed for formatting
            temperature=0.3,
            max_tokens=2048,
        )
        response_text: str = result["choices"][0]["message"].get("content", "")
        return {
            "response": response_text or "I was unable to generate a response from the data.",
            "error": None,
        }
    except Exception as exc:
        # Fallback: return a basic summary without LLM
        fallback = [
            f"Query returned {tr.get('row_count', 0)} row(s)."
            if tr.get("row_count", 0) > 0
            else "Query returned no results."
            for tr in tool_results
        ]
        return {
            "response": (
                "The database query completed but I encountered an error while formatting the response. "
                f"Here is the raw summary:\n{chr(10).join(fallback)}\n\n"
                f"Error: {exc}"
            ),
            "error": str(exc),
        }