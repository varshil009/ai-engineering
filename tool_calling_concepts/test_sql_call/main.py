"""Test Runner — evaluates LLM tool-calling accuracy on 20 predefined SQL queries.

Usage:
    python -m tool_calling_concepts.test_sql_call.main

This test framework:
    1. Loads 20 test queries with natural language prompts and expected SQL.
    2. For each query:
       a. Sends the prompt through the LLM agent (which calls the tool).
       b. Intercepts the tool call to capture the LLM-generated SQL.
       c. Executes the expected SQL directly against Supabase.
       d. Computes Query Match Score and Result Match Score.
    3. Aggregates results into a final report (console + JSON file).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

# Add project root to path so tool_calling_concepts and supabase_client are importable
_current_dir = os.path.dirname(os.path.abspath(__file__))
# tool_calling_concepts is at: ai-engineering/tool_calling_concepts/
# __file__ is at:            ai-engineering/tool_calling_concepts/test_sql_call/main.py
# So parent of _current_dir gives us the ai-engineering directory
_ai_engineering_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _ai_engineering_root not in sys.path:
    sys.path.insert(0, _ai_engineering_root)

# Also add the project root for .env and supabase_client
_project_root = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv

# Load .env before anything else
_dotenv_path = os.path.join(_project_root, ".env")
load_dotenv(_dotenv_path)

from supabase_client.client import SupabaseClient
from supabase_client.config import SupabaseConfig
from tool_calling_concepts.main import run_agent
from tool_calling_concepts.test_sql_call.metrics import (
    compute_query_match_score,
    compute_result_match_score,
)
from tool_calling_concepts.test_sql_call.test_queries import TEST_QUERIES


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _execute_sql_directly(sql: str) -> list[dict[str, Any]]:
    """Execute a raw SQL query directly against Supabase.

    Uses the REST-based Supabase client's _request_json method
    to send the query as a raw SQL command.
    """
    config = SupabaseConfig()
    if not config.is_configured():
        raise RuntimeError("Supabase is not configured. Check your .env file.")

    # Use the Supabase REST API's /rpc/ endpoint for raw SQL if available,
    # otherwise parse the SQL into REST params like the existing client does.
    from supabase_client.crud import SupabaseCRUD

    crud = SupabaseCRUD(config)
    # The crud._request_json can send GET requests to the table URL with params.
    # For arbitrary SQL, we need to parse the SELECT and build params.
    # We reuse the parse logic from services/supabase_client.py
    from tool_calling_concepts.services.supabase_client import _parse_select

    parsed = _parse_select(sql)
    table = parsed["table"]
    if not table:
        raise ValueError(f"Cannot parse table from SQL: {sql}")

    url = config.table_url(table)
    params: dict[str, str] = {
        "select": parsed.get("select", "*"),
    }
    # WHERE filters
    for col, val in parsed.get("filters", {}).items():
        params[col] = val
    # ORDER BY
    if parsed.get("order_by"):
        params["order"] = parsed["order_by"]
    # LIMIT
    if parsed.get("limit") is not None:
        params["limit"] = str(parsed["limit"])

    response_data = crud._request_json("GET", url, params=params)
    if response_data is None:
        return []
    return response_data


def _get_llm_sql_from_state(state: dict[str, Any]) -> str | None:
    """Extract the LLM-generated SQL from the intercepted state.

    The state after tool_executor node has tool_results containing the SQL used.
    """
    tool_results = state.get("tool_results", [])
    for tr in tool_results:
        sql = tr.get("sql_query")
        if sql:
            return sql
    return None


def _get_tool_error_from_state(state: dict[str, Any]) -> str | None:
    """Check if the tool execution returned an error."""
    tool_results = state.get("tool_results", [])
    for tr in tool_results:
        content = tr.get("content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "error" in parsed:
                    return parsed["error"]
            except (json.JSONDecodeError, TypeError):
                pass
    return state.get("error")


# ──────────────────────────────────────────────
# Individual test runner
# ──────────────────────────────────────────────


async def run_single_test(test_case: dict[str, Any]) -> dict[str, Any]:
    """Run a single test case through the LLM agent and evaluate.

    Returns a dict with all scores and metadata.
    """
    test_id = test_case["id"]
    prompt = test_case["prompt"]
    expected_sql = test_case["expected_sql"]
    description = test_case["description"]

    print(f"\n{'='*70}")
    print(f"[TEST #{test_id}] {description}")
    print(f"[PROMPT] {prompt[:120]}...")
    print(f"{'='*70}")

    # Store intercepted state
    intercepted_state: dict[str, Any] | None = None

    async def _intercept(state: dict[str, Any]) -> None:
        nonlocal intercepted_state
        intercepted_state = state
        llm_sql = _get_llm_sql_from_state(state)
        if llm_sql:
            print(f"  [LLM SQL] {llm_sql[:150]}...")
        else:
            print("  [WARN] No SQL generated by LLM")

    # Run the agent with interception
    start_time = time.time()
    try:
        final_state = await run_agent(
            query=prompt,
            intercept_callback=_intercept,
        )
        elapsed = time.time() - start_time
        print(f"  [TIME] {elapsed:.2f}s")
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"  [ERROR] Agent error: {exc}")
        return {
            "id": test_id,
            "prompt": prompt,
            "expected_sql": expected_sql,
            "description": description,
            "error": str(exc),
            "elapsed_seconds": round(elapsed, 2),
            "llm_generated_sql": None,
            "query_match_score": None,
            "result_match_score": None,
            "overall_score": 0.0,
            "tool_error": str(exc),
        }

    # Extract LLM-generated SQL from intercepted state
    llm_generated_sql = None
    if intercepted_state:
        llm_generated_sql = _get_llm_sql_from_state(intercepted_state)

    tool_error = _get_tool_error_from_state(intercepted_state or {})

    # Compute Query Match Score
    query_match = None
    if llm_generated_sql:
        query_match = compute_query_match_score(expected_sql, llm_generated_sql)
        print(f"  [QUERY MATCH] Score: {query_match['score']:.4f}")
        print(f"     -> {query_match['details']}")

    # Compute Result Match Score by running both queries directly
    result_match = None
    expected_results: list[dict[str, Any]] = []
    actual_results: list[dict[str, Any]] = []

    try:
        expected_results = _execute_sql_directly(expected_sql)
        print(f"  [EXPECTED] {len(expected_results)} rows returned")
    except Exception as exc:
        print(f"  [WARN] Expected SQL execution failed: {exc}")

    if llm_generated_sql:
        try:
            actual_results = _execute_sql_directly(llm_generated_sql)
            print(f"  [ACTUAL] {len(actual_results)} rows returned")
        except Exception as exc:
            print(f"  [WARN] LLM SQL execution failed: {exc}")

    # Compute result match
    if expected_results or actual_results:
        result_match = compute_result_match_score(expected_results, actual_results)
        print(f"  [RESULT MATCH] Score: {result_match['score']:.4f}")
        print(f"     -> {result_match['details']}")

    # Compute overall
    q_score = query_match["score"] if query_match else 0.0
    r_score = result_match["score"] if result_match else 0.0
    overall_score = q_score * 0.5 + r_score * 0.5

    return {
        "id": test_id,
        "prompt": prompt,
        "expected_sql": expected_sql,
        "description": description,
        "elapsed_seconds": round(elapsed, 2),
        "llm_generated_sql": llm_generated_sql,
        "tool_error": tool_error,
        "query_match_score": query_match,
        "result_match_score": result_match,
        "overall_score": round(overall_score, 4),
    }


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────


def _generate_report(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate individual test results into a comprehensive report."""
    total = len(all_results)
    completed = [r for r in all_results if r["overall_score"] is not None]
    errors = [r for r in all_results if r.get("error")]

    avg_query_score = (
        sum(r["query_match_score"]["score"] for r in completed if r["query_match_score"])
        / max(len([r for r in completed if r["query_match_score"]]), 1)
    )
    avg_result_score = (
        sum(r["result_match_score"]["score"] for r in completed if r["result_match_score"])
        / max(len([r for r in completed if r["result_match_score"]]), 1)
    )
    avg_overall = (
        sum(r["overall_score"] for r in completed if r["overall_score"] is not None)
        / max(len([r for r in completed if r["overall_score"] is not None]), 1)
    )

    # Categorize
    excellent = [r for r in completed if r["overall_score"] is not None and r["overall_score"] >= 0.90]
    good = [r for r in completed if r["overall_score"] is not None and 0.75 <= r["overall_score"] < 0.90]
    moderate = [r for r in completed if r["overall_score"] is not None and 0.50 <= r["overall_score"] < 0.75]
    poor = [r for r in completed if r["overall_score"] is not None and r["overall_score"] < 0.50]

    report = {
        "test_timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_tests": total,
            "completed": len(completed),
            "errors": len(errors),
            "average_query_match_score": round(avg_query_score, 4),
            "average_result_match_score": round(avg_result_score, 4),
            "average_overall_score": round(avg_overall, 4),
            "categorization": {
                "excellent (>=0.90)": len(excellent),
                "good (0.75-0.89)": len(good),
                "moderate (0.50-0.74)": len(moderate),
                "poor (<0.50)": len(poor),
            },
        },
        "results": all_results,
    }

    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print a human-readable report to the console."""
    summary = report["summary"]

    print("\n\n" + "=" * 70)
    print("  TOOL CALLING TEST REPORT")
    print(f"   Timestamp: {report['test_timestamp']}")
    print("=" * 70)

    print(f"\n  SUMMARY")
    print(f"   Total Tests:     {summary['total_tests']}")
    print(f"   Completed:       {summary['completed']}")
    print(f"   Errors:          {summary['errors']}")
    print(f"   -----------------------------------------")
    print(f"   Avg Query Score:  {summary['average_query_match_score']:.4f}")
    print(f"   Avg Result Score: {summary['average_result_match_score']:.4f}")
    print(f"   Avg Overall Score:{summary['average_overall_score']:.4f}")
    print(f"   -----------------------------------------")
    print(f"   Categorization:")
    for cat, count in summary["categorization"].items():
        bar = "#" * count + "." * max(0, summary["total_tests"] - count)
        print(f"     {cat:25s}: {count:2d} {bar}")

    print(f"\n  DETAILED RESULTS")
    for result in report["results"]:
        os_ = result["overall_score"]
        if os_ is not None and os_ >= 0.75:
            status = "[PASS]"
        elif os_ is not None:
            status = "[WARN]"
        else:
            status = "[FAIL]"
        print(f"\n   {status} Test #{result['id']}: {result['description'][:60]}")
        print(f"      Overall: {result['overall_score']:.4f}")
        if result["query_match_score"]:
            print(f"      Query:   {result['query_match_score']['score']:.4f}")
        if result["result_match_score"]:
            print(f"      Result:  {result['result_match_score']['score']:.4f}")
        if result.get("error"):
            print(f"      Error:   {result['error'][:100]}")
        if result.get("tool_error"):
            print(f"      Tool Err:{result['tool_error'][:100]}")
        if result.get("llm_generated_sql"):
            print(f"      LLM SQL: {result['llm_generated_sql'][:120]}...")

    print("\n" + "=" * 70)
    print("  REPORT COMPLETE")
    print("=" * 70)


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────


async def main() -> None:
    """Run all test queries and generate a report."""
    print("  Tool Calling Test Framework")
    print(f"   {len(TEST_QUERIES)} test queries loaded")
    print(f"   Started at: {datetime.now(timezone.utc).isoformat()}")

    all_results: list[dict[str, Any]] = []
    for test_case in TEST_QUERIES:
        result = await run_single_test(test_case)
        all_results.append(result)

        # Save intermediate results in case of crash
        _save_results(all_results, "partial")

    # Generate final report
    report = _generate_report(all_results)

    # Print to console
    _print_report(report)

    # Save to file
    _save_results(report, "final")

    print(f"\n  Full report saved to: test_report.json")
    print("  All tests completed.")


def _save_results(data: dict[str, Any] | list[dict[str, Any]], suffix: str) -> None:
    """Save results to a JSON file in the current directory."""
    filename = f"test_report_{suffix}.json"
    filepath = os.path.join(_current_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())