"""Async Supabase client using the REST API directly.

Translates SQL SELECT queries into Supabase REST API calls.
Pattern matches the proven ``supabase_client/`` package.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from tool_calling_concepts.config import settings


# ── SQL Parser ──────────────────────────────────────────────
def _parse_select(sql: str) -> dict[str, Any]:
    """Parse a SELECT query into Supabase REST API parameters.

    Supports: ``SELECT ... FROM table``, ``WHERE col = value``,
    ``ORDER BY col [ASC|DESC]``, ``LIMIT N``.

    Returns a dict with keys: table, select, filters, order_by, limit.
    """
    result: dict[str, Any] = {
        "table": None,
        "select": "*",
        "filters": {},
        "order_by": None,
        "limit": None,
    }

    sql_stripped = sql.strip().rstrip(";")
    upper = sql_stripped.upper()

    # Extract FROM clause
    from_match = re.search(r"\bFROM\b\s+(\w+)", sql_stripped, re.IGNORECASE)
    if not from_match:
        raise ValueError(f"Cannot parse table name from: {sql!r}")
    result["table"] = from_match.group(1)

    # Extract SELECT columns
    select_match = re.match(
        r"SELECT\s+(.+?)\s+FROM", sql_stripped, re.IGNORECASE | re.DOTALL
    )
    if select_match:
        cols = select_match.group(1).strip()
        if cols != "*":
            result["select"] = ",".join(c.strip().strip('"') for c in cols.split(","))

    # Extract WHERE conditions
    where_match = re.search(
        r"WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|\s*$)",
        sql_stripped,
        re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        where_text = where_match.group(1).strip()
        # Split on AND (ignoring case)
        parts = re.split(r"\s+AND\s+", where_text, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            # Match: column operator value
            m = re.match(r"(\w+)\s*([=<>!]+)\s*(.+)", part)
            if m:
                col = m.group(1).strip('"')
                op = m.group(2).strip()
                val = m.group(3).strip().strip("'\"")
                # Map SQL operators to Supabase REST operators
                op_map = {
                    "=": "eq",
                    "!=": "neq",
                    "<>": "neq",
                    ">": "gt",
                    ">=": "gte",
                    "<": "lt",
                    "<=": "lte",
                    "LIKE": "like",
                    "ILIKE": "ilike",
                }
                # Handle special case: value might be a function like NOW()
                if val.upper().startswith("NOW()"):
                    # Can't translate function calls - skip this filter
                    continue
                op_key = op_map.get(op.upper(), "eq")
                result["filters"][col] = f"{op_key}.{val}"

    # Extract ORDER BY
    order_match = re.search(
        r"ORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s*$)",
        sql_stripped,
        re.IGNORECASE | re.DOTALL,
    )
    if order_match:
        order_text = order_match.group(1).strip()
        order_parts = []
        for part in order_text.split(","):
            part = part.strip()
            dir = "asc"
            if " DESC" in part.upper():
                dir = "desc"
                part = re.sub(r"\s+DESC", "", part, flags=re.IGNORECASE).strip()
            elif " ASC" in part.upper():
                part = re.sub(r"\s+ASC", "", part, flags=re.IGNORECASE).strip()
            order_parts.append(f"{part}.{dir}")
        result["order_by"] = ",".join(order_parts)

    # Extract LIMIT
    limit_match = re.search(r"LIMIT\s+(\d+)", sql_stripped, re.IGNORECASE)
    if limit_match:
        result["limit"] = int(limit_match.group(1))

    return result


# ── Client ──────────────────────────────────────────────────


class SupabaseClient:
    """Async Supabase REST API client.

    Translates SQL SELECT queries into REST API GET requests.
    Uses service_role key for full read access.

    Usage::

        async with SupabaseClient() as client:
            rows = await client.execute_sql("SELECT * FROM SGT1 LIMIT 5")
    """

    def __init__(self) -> None:
        base_url = settings.supabase_url.rstrip("/")
        self._rest_url: str = f"{base_url}/rest/v1"
        self._headers: dict[str, str] = {
            "apikey": settings.supabase_service_role_key,
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SupabaseClient:
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def execute_sql(self, sql_query: str) -> list[dict[str, Any]]:
        """Execute a SELECT query via the Supabase REST API.

        Parses the SQL and translates it to REST API parameters.
        """
        if self._client is None:
            raise RuntimeError(
                "SupabaseClient must be used as an async context manager"
            )

        parsed = _parse_select(sql_query)
        #print("Supabase Client _parse_select:", parsed)
        table = parsed["table"]
        #print("Supabase Client: parsed['table']:", table)
        if not table:
            raise ValueError(f"Could not parse table from: {sql_query!r}")

        url = f"{self._rest_url}/{table}"
        params: dict[str, str] = {}

        # SELECT columns
        params["select"] = parsed["select"]

        # Filters (WHERE)
        for col, val in parsed["filters"].items():
            params[col] = val

        # ORDER BY
        if parsed.get("order_by"):
            params["order"] = parsed["order_by"]

        # LIMIT
        if parsed.get("limit") is not None:
            params["limit"] = str(parsed["limit"])
        print("Supabase Client: execute_sql params:", params)

        try:
            response = await self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Supabase API request timed out for table '{table}': {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Supabase API request failed for table '{table}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error during Supabase API request for table '{table}': {exc}"
            ) from exc

        try:
            if response.status_code >= 400:
                raise RuntimeError(
                    f"GET {url} failed with {response.status_code}: {response.text[:500]}"
                )
        except RuntimeError:
            raise  # Re-raise our own RuntimeError
        except Exception as exc:
            raise RuntimeError(
                f"Failed to check Supabase response status for table '{table}': {exc}"
            ) from exc

        try:
            data: list[dict[str, Any]] = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse Supabase response JSON for table '{table}': {exc}"
            ) from exc

        print(f"Supabase Client: execute_sql returned {len(data)} rows")
        #print(f"Supabase Client: execute_sql returned table : \n{table}")
        return data

async def run_sql_query(sql_query: str) -> list[dict[str, Any]]:
    """Convenience: open client, run query, return results.
    So basically, 
    async with SupabaseClient() as client: this creates an async operation
    SupabaseClient.__init__ prepares headers and base URL for the REST API
    SupabaseClient.__aenter__ creates an httpx.AsyncClient for making requests
    SupabaseClient.execute_sql parses the SQL query and converts into REST API parameters, then makes the GET request
    SupabaseClient.__aexit__ closes the httpx.AsyncClient when done

    SupabaseClient implements the asynchronous context manager protocol by defining __aenter__ and __aexit__. 
    This allows it to be used with async with, automatically creating and cleaning up the underlying httpx.AsyncClient.
    """
    async with SupabaseClient() as client:
        return await client.execute_sql(sql_query)