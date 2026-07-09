# AI Engineering

A collection of experimental projects exploring LLM integration patterns — from structured output generation and streaming chat to tool-calling agents with database access.

## Projects

### 1. `tool_calling_concepts/` — LLM Tool-Calling Agent

A LangGraph-based agent that uses **Groq** (LLaMA 3.3 70B) to answer natural-language questions about power-grid data stored in **Supabase**.

**Architecture:**
```
User Query → LLM Inference → Tool Executor → Response Formatter → Answer
                │                  │
                │ (no tool call)   │ (SQL results)
                └──→ Direct Answer ┘
```

**Key Components:**
- **`main.py`** — LangGraph orchestration with 3 nodes (LLM inference, tool execution, response formatting). Supports an optional `intercept_callback` for test frameworks.
- **`models/tools.py`** — Defines the `execute_supabase_query` function tool that the LLM can call.
- **`models/schemas.py`** — Pydantic/TypedDict schemas for the agent state.
- **`nodes/llm_inference.py`** — Sends the user query + system prompt (with full table schemas) to Groq, which decides whether to call the SQL tool or answer directly.
- **`nodes/tool_executor.py`** — Validates and executes the LLM-generated SQL against Supabase.
- **`nodes/response_formatter.py`** — Sends tool results back to the LLM for natural-language formatting.
- **`services/groq_client.py`** — Async wrapper around the Groq SDK.
- **`services/supabase_client.py`** — Translates SQL SELECT queries into Supabase REST API calls.
- **`utils/sql_validator.py`** — Blocks dangerous SQL (INSERT, UPDATE, DELETE, DROP, etc.).
- **`config.py`** — Table schema definitions (BOLNEY, SGT1–SGT4) and environment-based settings.

**Database Tables:**
| Table | Description |
|-------|-------------|
| `BOLNEY` | Actual active power readings with timestamps |
| `SGT1`–`SGT4` | Prediction data (actual vs predicted values, errors, forecast metadata) |

**Usage:**
```bash
python -m tool_calling_concepts.main "Show me the latest predictions from SGT1"
```

#### `test_sql_call/` — Tool-Calling Evaluation Framework

A test suite that evaluates how accurately the LLM generates SQL queries from natural language prompts.

**How it works:**
1. 20 predefined test queries cover various SQL patterns (WHERE, JOIN, GROUP BY, subqueries, aggregations, etc.)
2. For each query, the natural language prompt is sent through the agent
3. The LLM-generated SQL is **intercepted** after tool execution
4. Both the expected SQL and LLM-generated SQL are executed directly against Supabase
5. Two metrics are computed:
   - **Query Match Score** (0–1): Token-based structural similarity using LCS with weighted keywords
   - **Result Match Score** (0–1): Row-level set comparison (70% row overlap + 30% count proximity)
6. A comprehensive report is generated (console + JSON)

**Usage:**
```bash
python -m tool_calling_concepts.test_sql_call.main
```

**Files:**
- `test_queries.py` — 20 test cases with prompts and expected SQL
- `metrics.py` — Scoring functions for query and result comparison
- `main.py` — Test runner with interception and report generation

---

### 2. `ollama_chat/` — SSE Streaming Chat with Ollama

A **FastAPI** server that provides a chat interface to locally-hosted Ollama models with Server-Sent Events (SSE) streaming.

**Features:**
- Two streaming strategies:
  - **Accumulate & Flush** (`/chat/accumulate`): Buffers tokens and flushes in batches
  - **Direct Yield** (`/chat/direct`): Pushes every token immediately
- Client-side stop detection via `request.is_disconnected()`
- Detailed logging to both console and file (`sse_streaming.log`)
- Static HTML frontend served at `/`

**Usage:**
```bash
cd ollama_chat
pip install -r requirements.txt
uvicorn app:app --reload
```

**Dependencies:** `fastapi`, `uvicorn`, `ollama`

---

### 3. `constrained_json/` — Structured JSON Generation with Outlines

A minimal example of using the **Outlines** library with **Ollama** to generate structured JSON output constrained by a Pydantic schema.

**How it works:**
- Defines a `Profile` Pydantic model (`name: str`, `age: int`)
- Uses Outlines' generator to constrain the LLM output to match the schema
- The LLM (llama3.2:3b) generates a random person profile as valid JSON

**Usage:**
```bash
cd constrained_json
python main.py
```

**Dependencies:** `ollama`, `outlines`, `pydantic`

---

### 4. `DecoderConstraints.ipynb` — Jupyter Notebook

A Colab notebook exploring decoder-side constraints for structured generation. Contains experiments with GPU-accelerated inference and constrained decoding techniques.

---

## Shared Infrastructure

### `supabase_client/` (project root)

A reusable Supabase client package used by `tool_calling_concepts/`:
- **`config.py`** — Configuration from environment variables
- **`client.py`** — Unified client combining config and CRUD
- **`crud.py`** — CRUD operations (fetch, insert, update, delete) via REST API

### Environment Configuration

All projects share a `.env` file at the project root with:
- `GROQ_API_KEY` / `GROQ_MODEL` — For Groq-based projects
- `SUPABASE_PROJECT_URL` / `SUPABASE_SERVICE_ROLE_API` — For Supabase access
- `UKPN_API` / `UKPN_BASE_URL` — For UKPN data source