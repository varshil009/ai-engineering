"""20 Test SQL Queries with natural language prompts and expected SQL.

Each entry contains:
    - id: Unique identifier
    - prompt: Natural language query (what the user would ask)
    - expected_sql: The correct SQL query to answer the prompt
    - description: What skill/capability this query tests
"""

from typing import Any

TEST_QUERIES: list[dict[str, Any]] = [
    # ── 1. Simple SELECT with WHERE ──────────────────────────────────────────
    {
        "id": 1,
        "prompt": "Show me all records from BOLNEY where the SGT is 'SGT1'.",
        "expected_sql": "SELECT * FROM BOLNEY WHERE SGT = 'SGT1'",
        "description": "Simple SELECT with WHERE equality filter on a text column",
    },
    # ── 2. SELECT with multiple AND conditions ──────────────────────────────
    {
        "id": 2,
        "prompt": "Get all SGT1 predictions where the actual value is greater than 100 and the absolute percentage error is less than 5.",
        "expected_sql": "SELECT * FROM SGT1 WHERE actual > 100 AND absolute_percentage_error < 5",
        "description": "Multiple WHERE conditions with AND, numeric comparisons",
    },
    # ── 3. COUNT with WHERE ─────────────────────────────────────────────────
    {
        "id": 3,
        "prompt": "How many records are there in the BOLNEY table?",
        "expected_sql": "SELECT COUNT(*) FROM BOLNEY",
        "description": "Simple COUNT aggregation without filter",
    },
    # ── 4. ORDER BY with LIMIT ──────────────────────────────────────────────
    {
        "id": 4,
        "prompt": "Show me the top 5 most recent records from the BOLNEY table based on the Date column.",
        "expected_sql": "SELECT * FROM BOLNEY ORDER BY Date DESC LIMIT 5",
        "description": "ORDER BY DESC with LIMIT for sorting",
    },
    # ── 5. Column selection with WHERE ──────────────────────────────────────
    {
        "id": 5,
        "prompt": "Get the predicted_for_utc and predicted values from SGT2 where the forecast_horizon_minutes is 60.",
        "expected_sql": "SELECT predicted_for_utc, predicted FROM SGT2 WHERE forecast_horizon_minutes = 60",
        "description": "Specific column selection with WHERE on numeric column",
    },
    # ── 6. GROUP BY with COUNT ──────────────────────────────────────────────
    {
        "id": 6,
        "prompt": "How many records exist for each SGT in the BOLNEY table?",
        "expected_sql": "SELECT SGT, COUNT(*) FROM BOLNEY GROUP BY SGT",
        "description": "GROUP BY with COUNT aggregation",
    },
    # ── 7. GROUP BY with AVG ────────────────────────────────────────────────
    {
        "id": 7,
        "prompt": "What is the average ActivePower_Avg for each SGT in the BOLNEY table?",
        "expected_sql": "SELECT SGT, AVG(ActivePower_Avg) FROM BOLNEY GROUP BY SGT",
        "description": "GROUP BY with AVG aggregation",
    },
    # ── 8. GROUP BY with SUM ────────────────────────────────────────────────
    {
        "id": 8,
        "prompt": "What is the total sum of actual values for each prediction_index in SGT1?",
        "expected_sql": "SELECT prediction_index, SUM(actual) FROM SGT1 GROUP BY prediction_index",
        "description": "GROUP BY with SUM aggregation",
    },
    # ── 9. MIN and MAX aggregation ──────────────────────────────────────────
    {
        "id": 9,
        "prompt": "What are the minimum and maximum actual values in the SGT3 table?",
        "expected_sql": "SELECT MIN(actual), MAX(actual) FROM SGT3",
        "description": "MIN and MAX aggregation functions",
    },
    # ── 10. WHERE with BETWEEN ──────────────────────────────────────────────
    {
        "id": 10,
        "prompt": "Find all BOLNEY records where ActivePower_Avg is between 50 and 200.",
        "expected_sql": "SELECT * FROM BOLNEY WHERE ActivePower_Avg BETWEEN 50 AND 200",
        "description": "WHERE with BETWEEN range filter",
    },
    # ── 11. WHERE with IN clause ────────────────────────────────────────────
    {
        "id": 11,
        "prompt": "Get all records from BOLNEY where SGT is either 'SGT1' or 'SGT2'.",
        "expected_sql": "SELECT * FROM BOLNEY WHERE SGT IN ('SGT1', 'SGT2')",
        "description": "WHERE with IN list filter",
    },
    # ── 12. WHERE with NOT EQUAL ────────────────────────────────────────────
    {
        "id": 12,
        "prompt": "Show all SGT4 records where the residual is not equal to 0.",
        "expected_sql": "SELECT * FROM SGT4 WHERE residual != 0",
        "description": "WHERE with not-equal operator",
    },
    # ── 13. ORDER BY with multiple columns ──────────────────────────────────
    {
        "id": 13,
        "prompt": "List all SGT1 predictions sorted by prediction_index ascending and then by predicted_for_utc descending.",
        "expected_sql": "SELECT * FROM SGT1 ORDER BY prediction_index ASC, predicted_for_utc DESC",
        "description": "ORDER BY with multiple columns and mixed directions",
    },
    # ── 14. GROUP BY with HAVING ────────────────────────────────────────────
    {
        "id": 14,
        "prompt": "Show me SGT values from BOLNEY that have more than 100 records.",
        "expected_sql": "SELECT SGT, COUNT(*) FROM BOLNEY GROUP BY SGT HAVING COUNT(*) > 100",
        "description": "GROUP BY with HAVING filter on aggregation",
    },
    # ── 15. Subquery ────────────────────────────────────────────────────────
    {
        "id": 15,
        "prompt": "Find all BOLNEY records where ActivePower_Avg is above the average ActivePower_Avg across all records.",
        "expected_sql": "SELECT * FROM BOLNEY WHERE ActivePower_Avg > (SELECT AVG(ActivePower_Avg) FROM BOLNEY)",
        "description": "Subquery with scalar comparison",
    },
    # ── 16. COUNT with DISTINCT ─────────────────────────────────────────────
    {
        "id": 16,
        "prompt": "How many unique SGT values are there in the BOLNEY table?",
        "expected_sql": "SELECT COUNT(DISTINCT SGT) FROM BOLNEY",
        "description": "COUNT with DISTINCT to count unique values",
    },
    # ── 17. WHERE with LIKE pattern matching ────────────────────────────────
    {
        "id": 17,
        "prompt": "Find all records in SGT2 where the source_file starts with 'prediction'.",
        "expected_sql": "SELECT * FROM SGT2 WHERE source_file LIKE 'prediction%'",
        "description": "WHERE with LIKE pattern matching on text column",
    },
    # ── 18. Complex aggregation with multiple functions ─────────────────────
    {
        "id": 18,
        "prompt": "For each SGT in BOLNEY, show the count, average, min, and max of ActivePower_Avg.",
        "expected_sql": "SELECT SGT, COUNT(*), AVG(ActivePower_Avg), MIN(ActivePower_Avg), MAX(ActivePower_Avg) FROM BOLNEY GROUP BY SGT",
        "description": "Complex aggregation with multiple functions in GROUP BY",
    },
    # ── 19. Date range query ────────────────────────────────────────────────
    {
        "id": 19,
        "prompt": "Show all BOLNEY records from the last 7 days.",
        "expected_sql": "SELECT * FROM BOLNEY WHERE Date >= NOW() - INTERVAL '7 days'",
        "description": "Date range query with NOW() and INTERVAL",
    },
    # ── 20. LIMIT with OFFSET-style (SELECT with WHERE, ORDER BY, LIMIT) ────
    {
        "id": 20,
        "prompt": "Show me the 10 most recent BOLNEY records where ActivePower_Avg is greater than 0, ordered by Date descending.",
        "expected_sql": "SELECT * FROM BOLNEY WHERE ActivePower_Avg > 0 ORDER BY Date DESC LIMIT 10",
        "description": "Combined WHERE, ORDER BY, and LIMIT",
    },
]