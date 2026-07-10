"""Terminal Executor Node — runs Python analysis code in a persistent subprocess.

The subprocess (``uv run python -i``) stays open for the lifetime of the
user session. Code is sent via stdin, output is captured from stdout.

Security: blocks dangerous commands like ``cd``, ``import os``, ``subprocess``,
``shutil``, ``eval``, ``exec``, and filesystem exploration.
"""

import asyncio
import json
import re
from typing import Any

from tool_calling_concepts.models.schemas import AgentState

# ── Blocked patterns ──────────────────────────────────────────────────
_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bcd\s+\.\."),
    re.compile(r"\bimport\s+os\b"),
    re.compile(r"\bimport\s+subprocess\b"),
    re.compile(r"\bimport\s+shutil\b"),
    re.compile(r"\bimport\s+pathlib\b"),
    re.compile(r"\bimport\s+sys\b"),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bos\.popen\b"),
    re.compile(r"\bsubprocess\.\w+"),
    re.compile(r"\bshutil\.\w+"),
    re.compile(r"\b__import__\b"),
    re.compile(r"\bglob\.glob\b"),
    re.compile(r"\bglob\.iglob\b"),
    re.compile(r"\bos\.listdir\b"),
    re.compile(r"\bos\.walk\b"),
    re.compile(r"\bpathlib\.\w+"),
]

# ── Persistent subprocess ──────────────────────────────────────────────

_process: asyncio.subprocess.Process | None = None


async def _get_process() -> asyncio.subprocess.Process:
    """Get or create the persistent Python subprocess."""
    global _process
    if _process is None or _process.returncode is not None:
        _process = await asyncio.create_subprocess_exec(
            "uv", "run", "python", "-i",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    return _process


async def close_terminal() -> None:
    """Close the persistent subprocess (call on client disconnect)."""
    global _process
    if _process is not None and _process.returncode is None:
        _process.stdin.write(b"exit()\n")
        await _process.stdin.drain()
        _process.stdin.close()
        await _process.wait()
    _process = None


def _validate_code(code: str) -> None:
    """Raise ValueError if code contains blocked patterns."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(code):
            raise ValueError(
                f"Blocked command detected matching pattern: {pattern.pattern}. "
                "Only data analysis code is allowed (no filesystem access, "
                "no subprocesses, no eval/exec)."
            )


async def _send_and_receive(code: str, timeout: float = 30.0) -> str:
    """Send code to the subprocess and read output until the next prompt.

    Args:
        code: Python code to execute.
        timeout: Maximum seconds to wait for output.

    Returns:
        Captured stdout from the subprocess.
    """
    proc = await _get_process()

    # Send the code
    proc.stdin.write(f"{code}\n".encode())
    proc.stdin.write(b'print("__PYTHON_END__")\n')
    await proc.stdin.drain()

    # Read until we see the sentinel
    output_lines: list[str] = []
    try:
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            if decoded.strip() == "__PYTHON_END__":
                break
            # Skip the Python prompt lines
            if decoded.strip() in (">>>", "...", ">>> ", "... "):
                continue
            output_lines.append(decoded)
    except asyncio.TimeoutError:
        output_lines.append(f"\n[ERROR: Execution timed out after {timeout}s]")

    return "\n".join(output_lines)


# ── LangGraph node ─────────────────────────────────────────────────────


async def terminal_executor_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: executes Python analysis code in the persistent terminal.

    Expects a tool call with name ``execute_python_analysis`` containing
    a ``code`` argument. Returns the stdout output as the tool result.

    Args:
        state: The current agent state with pending ``tool_calls``.

    Returns:
        Updated state with ``tool_results`` appended.
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

        if tool_name != "execute_python_analysis":
            continue  # Skip non-terminal tool calls

        # Parse arguments
        try:
            arguments: dict[str, Any] = json.loads(function_info.get("arguments", "{}"))
            code: str = arguments.get("code", "")
        except (json.JSONDecodeError, KeyError) as exc:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"Failed to parse tool arguments: {exc}"}),
            })
            last_error = str(exc)
            continue

        # Validate code
        try:
            _validate_code(code)
        except ValueError as exc:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"Code validation failed: {exc}"}),
            })
            last_error = str(exc)
            continue

        # Execute code
        try:
            output = await _send_and_receive(code)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": output,
                "terminal_output": True,
            })
        except Exception as exc:
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"error": f"Terminal execution failed: {exc}"}),
            })
            last_error = str(exc)

    return {
        "tool_results": tool_results,
        "error": last_error,
    }