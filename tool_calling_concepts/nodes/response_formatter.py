"""Response Formatter Node — takes tool results and generates a natural-language answer.

If the accumulated response would be too large (~6000 words), it returns a signal
telling the LLM to use the terminal tool for analysis instead.
"""

import json
from typing import Any

from tool_calling_concepts.models.schemas import AgentState
from tool_calling_concepts.services.groq_client import GroqClient

# If the combined messages exceed this word count, we'll force terminal analysis
_RESPONSE_WORD_LIMIT = 6000


def _estimate_word_count(data: list[dict[str, Any]]) -> int:
    """Estimate the word count of serialised data."""
    text = json.dumps(data, default=str)
    return len(text.split())


async def response_formatter_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: synthesises a final answer from tool results.

    Sends the original query, the SQL that was run, and the query results
    back to the LLM to generate a concise, natural-language response.

    If the data is too large, it injects a system instruction telling the
    LLM to use the terminal analysis tool instead of returning raw data.

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

    # Check if the data is too large — if so, signal back to loop with terminal
    total_words = sum(
        _estimate_word_count(rs.get("data", []))
        for rs in results_summary
    )

    if total_words > _RESPONSE_WORD_LIMIT:
        # Data is too large — return a special response that tells the LLM
        # to use the terminal analysis tool
        return {
            "response": None,
            "error": None,
            "_large_data": True,
            "_large_data_hint": (
                f"The query returned approximately {total_words} words of data, "
                f"which exceeds the {_RESPONSE_WORD_LIMIT} word limit. "
                "Use the `execute_python_analysis` tool to analyse this data. "
                "Write Python code to summarise it: count rows, get top 10, "
                "compute basic statistics (min, max, mean). "
                "Do NOT try to return the raw data in your response."
            ),
        }

    # Data is small enough — proceed with normal response formatting
    system_prompt = (
        "Generate a 2 line summary from data returned from the database query and format it into a markdown format."
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
        final_response = response_text or "I was unable to generate a response from the data."

        # Check if the response itself is too large
        if len(final_response.split()) > _RESPONSE_WORD_LIMIT:
            return {
                "response": None,
                "error": None,
                "_large_data": True,
                "_large_data_hint": (
                    f"The generated response is approximately {len(final_response.split())} words, "
                    f"which exceeds the {_RESPONSE_WORD_LIMIT} word limit. "
                    "Use the `execute_python_analysis` tool to analyse the data instead. "
                    "Write Python code to compute a concise summary."
                ),
            }

        return {
            "response": final_response,
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
