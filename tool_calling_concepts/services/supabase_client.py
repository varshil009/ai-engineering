"""Async Supabase client for executing read-only SQL queries.

Uses the Supabase REST API RPC endpoint to execute raw SQL.

Requires creating a PostgreSQL function in your Supabase project:

.. code-block:: sql

    CREATE OR REPLACE FUNCTION query(sql_query TEXT)
    RETURNS JSON
    LANGUAGE plpgsql
    SECURITY DEFINER
    AS $$
    DECLARE
      result JSON;
    BEGIN
      EXECUTE sql_query INTO result;
      RETURN result;
    END;
    $$;

Then call via: POST {SUPABASE_URL}/rest/v1/rpc/query
"""

from typing import Any

import httpx

from tool_calling_concepts.config import settings


class SupabaseClient:
    """Thin async wrapper around the Supabase REST API.

    Uses the service_role key for full read access to all tables.
    """

    def __init__(self) -> None:
        base_url = settings.supabase_url.rstrip("/")
        self._rpc_url: str = f"{base_url}/rest/v1/rpc/"
        self._headers: dict[str, str] = {
            "apikey": settings.supabase_service_role_key,
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SupabaseClient":
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def execute_sql(self, sql_query: str) -> list[dict[str, Any]]:
        """Execute a SELECT SQL query via the custom ``query`` RPC function.

        Args:
            sql_query: The SELECT SQL statement to execute.

        Returns:
            A list of row dicts.

        Raises:
            httpx.HTTPStatusError: If Supabase returns a non-2xx status
                (e.g. 404 if the RPC function doesn't exist).
            RuntimeError: If the client is not initialised.
        """
        if self._client is None:
            raise RuntimeError(
                "SupabaseClient must be used as an async context manager "
                "(async with SupabaseClient() as client: ...)"
            )

        response = await self._client.post(
            f"{self._rpc_url}query",
            json={"sql_query": sql_query},
        )
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json()
        return data


async def run_sql_query(sql_query: str) -> list[dict[str, Any]]:
    """Convenience function: open client, run query, return results."""
    async with SupabaseClient() as client:
        return await client.execute_sql(sql_query)