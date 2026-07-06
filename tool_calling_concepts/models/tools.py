"""Tool definitions for Groq function calling."""

from typing import Any


def get_execute_supabase_query_tool() -> dict[str, Any]:
    """Return the tool definition for executing Supabase SQL queries.

    This is passed to Groq as part of the `tools` parameter so the
    model knows it can call this function.
    """
    return {
        "type": "function",
        "function": {
            "name": "execute_supabase_query",
            "description": (
                "Execute a read-only SQL SELECT query on the Supabase PostgreSQL database. "
                "The database contains transformer/power-grid data across 4 SGT tables "
                "(SGT1, SGT2, SGT3, SGT4) and a BOLNEY table. "
                "Only SELECT queries are permitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": (
                            "The SQL SELECT query to execute. "
                            "Must be a valid PostgreSQL SELECT statement. "
                            "Use proper table names (BOLNEY, SGT1, SGT2, SGT3, SGT4) "
                            "and column names as defined in the schema."
                        ),
                    },
                },
                "required": ["sql_query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }