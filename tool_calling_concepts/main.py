"""Tool Calling System — LangGraph orchestration for Groq + Supabase.

Usage:
    python -m tool_calling_concepts.main "Show me the latest predictions from SGT1"

This builds a 3-node LangGraph:
    1. LLM Inference Node  — Groq decides tool call or direct answer
    2. Tool Executor Node  — Runs validated SQL on Supabase
    3. Response Formatter  — Synthesises natural-language answer from results
"""

import asyncio
import os
import sys
from collections.abc import Callable, Coroutine
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from tool_calling_concepts.config import settings
from tool_calling_concepts.models.schemas import AgentState
from tool_calling_concepts.nodes.llm_inference import llm_inference_node
from tool_calling_concepts.nodes.response_formatter import response_formatter_node
from tool_calling_concepts.nodes.tool_executor import tool_executor_node

# ──────────────────────────────────────────────
# Load environment variables from project root .env
# ──────────────────────────────────────────────

# Walk up from this file's directory to find the .env at project root
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
_dotenv_path = os.path.join(_project_root, ".env")
load_dotenv(_dotenv_path)

# Re-load settings after dotenv
# (settings singleton needs to be re-initialised since it was imported
#  before dotenv ran — we refresh it here)
from tool_calling_concepts.config import Settings as _Settings  # noqa: E402

# Re-create settings with fresh env values — assign into module
import tool_calling_concepts.config as _config_mod  # noqa: E402

_config_mod.settings = _Settings(
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    supabase_url=os.getenv("SUPABASE_PROJECT_URL", ""),
    supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_API", ""),
)


# ──────────────────────────────────────────────
# Conditional edge logic
# ──────────────────────────────────────────────


def _has_tool_calls(state: AgentState) -> str:
    """Decide which node to go to next based on state."""
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        return "tool_executor"
    return "end"


# ──────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────


def build_agent_graph() -> StateGraph:
    """Construct the LangGraph state machine.

    Graph topology::

        [START] → llm_inference → (tool_calls?)
                ├── yes → tool_executor → response_formatter → [END]
                └── no  → [END]
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("llm_inference", llm_inference_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("response_formatter", response_formatter_node)

    # Edges
    builder.set_entry_point("llm_inference")

    # Conditional: after LLM, check if we need to run tools
    builder.add_conditional_edges(
        "llm_inference",
        _has_tool_calls,
        {
            "tool_executor": "tool_executor",
            "end": END,
        },
    )

    # After tool execution, always format the response
    builder.add_edge("tool_executor", "response_formatter")

    # Response formatter ends the graph
    builder.add_edge("response_formatter", END)

    return builder


# ──────────────────────────────────────────────
# Compile & run
# ──────────────────────────────────────────────


async def run_agent(
    query: str,
    intercept_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
) -> dict[str, Any]:
    """Run the agent graph with a user query.

    Args:
        query: The natural-language query to process.
        intercept_callback: Optional async callback invoked after tool execution
            with the intermediate state (containing tool_results). The callback
            receives the state dict and can inspect/modify it before the graph
            continues to response formatting.

    Returns:
        The final state dict with keys: query, messages, response, error, etc.
    """
    settings.validate()

    graph = build_agent_graph()
    compiled = graph.compile()

    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "tool_calls": [],
        "tool_results": [],
        "sql_query": None,
        "response": None,
        "error": None,
    }

    if intercept_callback:
        # Run step-by-step so we can intercept after tool execution.
        # stream_mode="values" yields the full state at each step.
        final_state: AgentState = initial_state
        async for step in compiled.astream(initial_state, stream_mode="values"):
            final_state = step
            # After tool_executor node runs, its output is stored under the key
            # of the node name in the state. We detect this by checking for
            # tool_results in the state (which tool_executor populates).
            if step.get("tool_results"):
                await intercept_callback(dict(step))
        return dict(final_state)
    else:
        result: AgentState = await compiled.ainvoke(initial_state)
        return dict(result)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────


def main() -> None:
    """CLI entry point: reads query from first argument and prints results."""
    if len(sys.argv) < 2:
        print("Usage: python -m tool_calling_concepts.main \"<your query>\"", file=sys.stderr)
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()
    print(f"\n🔍 Query: {query}\n")
    print("─" * 60)

    try:
        result = asyncio.run(run_agent(query))
    except Exception as exc:
        print(f"\n❌ Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)

    print()

    # Display response
    response = result.get("response")
    error = result.get("error")

    if response:
        print(f"💬 Response:\n{response}")
    elif error:
        print(f"❌ Error: {error}")

    # Show SQL that was run (if any)
    tool_results = result.get("tool_results", [])
    for tr in tool_results:
        sql = tr.get("sql_query")
        row_count = tr.get("row_count")
        if sql:
            print(f"\n📊 SQL: {sql}")
            if row_count is not None:
                print(f"   Rows returned: {row_count}")

    print("\n" + "─" * 60)
    print("✅ Done.")


if __name__ == "__main__":
    main()