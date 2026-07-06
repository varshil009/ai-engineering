"""SQL query validator — ensures only safe SELECT queries are executed."""

import re
from typing import Final

# ──────────────────────────────────────────────
# Blocked patterns — any match → reject
# ──────────────────────────────────────────────

_BLOCKED_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bINSERT\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\b", re.IGNORECASE),
    re.compile(r"\bDELETE\b", re.IGNORECASE),
    re.compile(r"\bDROP\b", re.IGNORECASE),
    re.compile(r"\bALTER\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
    re.compile(r"\bCREATE\b", re.IGNORECASE),
    re.compile(r"\bREPLACE\b", re.IGNORECASE),
    re.compile(r"\bEXEC\b", re.IGNORECASE),
    re.compile(r"\bEXECUTE\b", re.IGNORECASE),
    re.compile(r"--"),  # SQL comment injection
    re.compile(r"/\*"),  # Block comment injection
    re.compile(r";\s*\w"),  # Multiple statements
]


def validate_sql_query(sql: str) -> str:
    """Validate and normalise a SQL query.

    Args:
        sql: The raw SQL string to validate.

    Returns:
        The trimmed, validated SQL string.

    Raises:
        ValueError: If the query contains blocked patterns or is empty.
    """
    stripped = sql.strip()

    if not stripped:
        raise ValueError("SQL query is empty.")

    # Must start with SELECT (case-insensitive, after stripping)
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise ValueError(
            "Only SELECT queries are allowed. "
            f"Query starts with: {stripped.split()[0] if stripped.split() else '<empty>'}"
        )

    # Check for blocked patterns
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(stripped):
            raise ValueError(
                f"SQL query contains blocked pattern: {pattern.pattern!r}"
            )

    return stripped