"""Tool Calling System — LangGraph orchestration for Groq + Supabase + Terminal.

Usage:
    python -m tool_calling_concepts.main "Show me the latest predictions from SGT1"

This builds a looping LangGraph:
    1. LLM Inference Node  — Groq decides tool call or direct answer
    2. Tool Executor Node  — Runs validated SQL on Supabase
    3. Terminal Executor Node — Runs Python analysis code in persistent subprocess
    4. Response Formatter  — Synthesises natural-language answer from results

The graph loops: llm_inference → tool_executor/terminal_executor → llm_inference
until the LLM responds directly (no tool calls), then goes to response_formatter.
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
from tool_calling_concepts.nodes.terminal_executor import (
    close_terminal,
    terminal_executor_node,
)
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


def _route_tool_calls(state: AgentState) -> str:
    """Route to the appropriate executor based on tool call type.

    Returns:
        "tool_executor" if SQL tool, "terminal_executor" if Python analysis tool,
        "response_formatter" if no tool calls.
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return "response_formatter"

    # Check the first tool call to determine routing
    for tc in tool_calls:
        fn_info = tc.get("function", {})
        name = fn_info.get("name", "")
        if name == "execute_python_analysis":
            return "terminal_executor"
        # Default to SQL executor for any other tool
        return "tool_executor"

    return "response_formatter"


def _should_loop_back(state: AgentState) -> str:
    """After tool execution, always loop back to LLM inference.

    The LLM will decide whether to call more tools or respond directly.
    """
    return "llm_inference"


def _after_formatter(state: AgentState) -> str:
    """After response formatter, either end or loop back for large data.

    If the response formatter set ``_large_data = True``, loop back to
    LLM inference so it can use the terminal tool.
    """
    if state.get("_large_data"):
        return "llm_inference"
    return "end"


# ──────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────


def build_agent_graph() -> StateGraph:
    """Construct the LangGraph state machine with looping.

    Graph topology::

        [START] → llm_inference → (tool_calls?)
                ├── SQL tool → tool_executor ──┐
                ├── Python tool → terminal_executor ──┐
                └── no tools → response_formatter ── (large data?)
                                    ├── yes → llm_inference (loop back)
                                    └── no  → [END]
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("llm_inference", llm_inference_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("terminal_executor", terminal_executor_node)
    builder.add_node("response_formatter", response_formatter_node)

    # Edges
    builder.set_entry_point("llm_inference")

    # After LLM: route to the right executor or end
    builder.add_conditional_edges(
        "llm_inference",
        _route_tool_calls,
        {
            "tool_executor": "tool_executor",
            "terminal_executor": "terminal_executor",
            "response_formatter": "response_formatter",
        },
    )

    # After either executor, loop back to LLM inference
    builder.add_conditional_edges(
        "tool_executor",
        _should_loop_back,
        {"llm_inference": "llm_inference"},
    )
    builder.add_conditional_edges(
        "terminal_executor",
        _should_loop_back,
        {"llm_inference": "llm_inference"},
    )

    # After formatter: either end or loop back for large data analysis
    builder.add_conditional_edges(
        "response_formatter",
        _after_formatter,
        {
            "llm_inference": "llm_inference",
            "end": END,
        },
    )

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
    import sys
    print(f"[DEBUG run_agent] Starting agent with query: {query[:80]}...", flush=True)
    sys.stdout.flush()
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

    try:
        if intercept_callback:
            # Run step-by-step so we can intercept after tool execution.
            # stream_mode="values" yields the full state at each step.
            final_state: AgentState = initial_state
            async for step in compiled.astream(initial_state, stream_mode="values"):
                final_state = step
                print(f"[DEBUG run_agent] Step keys: {list(step.keys())}", flush=True)
                sys.stdout.flush()
                if step.get("error"):
                    print(f"[DEBUG run_agent] State has error: {step['error']}", flush=True)
                    sys.stdout.flush()
                # Call intercept callback whenever we have tool_results OR an error
                if step.get("tool_results"):
                    print(f"[DEBUG run_agent] Calling intercept_callback (tool_results present)", flush=True)
                    sys.stdout.flush()
                    await intercept_callback(dict(step))
                elif step.get("error"):
                    # Also call intercept on errors so the test framework can see them
                    print(f"[DEBUG run_agent] Calling intercept_callback (error present)", flush=True)
                    sys.stdout.flush()
                    await intercept_callback(dict(step))
            print(f"[DEBUG run_agent] Final state keys: {list(final_state.keys())}", flush=True)
            print(f"[DEBUG run_agent] Final response: {str(final_state.get('response', ''))[:100]}", flush=True)
            print(f"[DEBUG run_agent] Final error: {final_state.get('error', 'None')}", flush=True)
            sys.stdout.flush()
            return dict(final_state)
        else:
            result: AgentState = await compiled.ainvoke(initial_state)
            print(f"[DEBUG run_agent] Invoke result keys: {list(result.keys())}", flush=True)
            print(f"[DEBUG run_agent] Invoke response: {str(result.get('response', ''))[:100]}", flush=True)
            print(f"[DEBUG run_agent] Invoke error: {result.get('error', 'None')}", flush=True)
            sys.stdout.flush()
            return dict(result)
    finally:
        # Clean up the persistent terminal subprocess
        print(f"[DEBUG run_agent] Cleaning up terminal", flush=True)
        sys.stdout.flush()
        await close_terminal()


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

    # Show tool results summary
    tool_results = result.get("tool_results", [])
    print(f"\n🛠️  Tool Results: {tool_results} entries")
    for tr in tool_results:
        sql = tr.get("sql_query")
        row_count = tr.get("row_count")
        if sql:
            print(f"\n📊 SQL: {sql}")
            if row_count is not None:
                print(f"   Rows returned: {row_count}")
        terminal_output = tr.get("terminal_output")
        if terminal_output:
            content = tr.get("content", "")
            # Show first 500 chars of terminal output
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(f"\n🐍 Python Analysis Output:\n{preview}")

    print("\n" + "─" * 60)
    print("✅ Done.")


if __name__ == "__main__":
    main()
