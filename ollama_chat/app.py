import logging
import sys
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import ollama
import asyncio
import json
import time
from pathlib import Path

# ==================== LOGGING SETUP ====================
logger = logging.getLogger("ollama_sse")
logger.setLevel(logging.DEBUG)

# Console handler - writes to terminal (stderr so it doesn't interfere with SSE)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# File handler - writes to a log file for review
file_handler = logging.FileHandler(Path(__file__).parent / "sse_streaming.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Suppress noisy uvicorn/starlette access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

logger.info("=" * 70)
logger.info("OLLAMA SSE STREAMING SERVER STARTED")
logger.info("=" * 70)

app = FastAPI(title="Ollama SSE Chat")


# --- SSE Streaming Strategy 1: Accumulate & Flush (Buffered) ---
async def stream_accumulate(request: Request, model: str, messages: list, flush_every_chars: int = 5):
    """
    Strategy 1: Accumulates chunks from Ollama and flushes them to the client
    in batches. Checks request.is_disconnected() to detect client stop.
    """
    buffer = ""
    accumulated = ""
    total_tokens_from_ollama = 0
    flush_count = 0
    start_time = time.time()
    stopped_by_client = False
    
    user_msg = messages[-1]["content"][:50] + ("..." if len(messages[-1]["content"]) > 50 else "")
    logger.info(f"[ACCUMULATE] ----------------------------------------------")
    logger.info(f"[ACCUMULATE] Starting stream | model={model} | flush_every={flush_every_chars} chars")
    logger.info(f"[ACCUMULATE] User message: \"{user_msg}\"")
    
    try:
        logger.info(f"[ACCUMULATE] Calling ollama.chat(model={model}, stream=True)...")
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        logger.info(f"[ACCUMULATE] Ollama stream opened successfully")
    except Exception as e:
        logger.error(f"[ACCUMULATE] FAILED to open Ollama stream: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Ollama error: {str(e)}'})}\n\n"
        return
    
    for chunk_idx, chunk in enumerate(stream):
        # Check if client disconnected (clicked Stop)
        if await request.is_disconnected():
            logger.info(f"[ACCUMULATE] Client disconnected - stopping stream")
            stopped_by_client = True
            # Explicitly close the Ollama stream generator to stop model generation
            # This tells Ollama's server to cancel the ongoing inference
            stream.close()
            logger.info(f"[ACCUMULATE] Ollama stream generator closed - model stopped")
            break
        
        token = chunk.get("message", {}).get("content", "")
        
        if token:
            total_tokens_from_ollama += 1
            buffer += token
            accumulated += token
            
            logger.debug(f"[ACCUMULATE] Token #{total_tokens_from_ollama}: repr={repr(token)} | buffer_len={len(buffer)} | accumulated_len={len(accumulated)}")
            
            # Flush when buffer reaches threshold
            if len(buffer) >= flush_every_chars:
                flush_count += 1
                logger.info(f"[ACCUMULATE] >>> FLUSH #{flush_count} | buffer=\"{buffer}\" | accumulated_len={len(accumulated)}")
                
                yield f"data: {json.dumps({'type': 'chunk', 'content': buffer, 'accumulated': accumulated})}\n\n"
                buffer = ""
                await asyncio.sleep(0)  # Yield control to event loop
    
    # Flush any remaining buffer (only if not stopped by client)
    if buffer and not stopped_by_client:
        flush_count += 1
        logger.info(f"[ACCUMULATE] >>> FINAL FLUSH #{flush_count} | buffer=\"{buffer}\" | accumulated_len={len(accumulated)}")
        yield f"data: {json.dumps({'type': 'chunk', 'content': buffer, 'accumulated': accumulated})}\n\n"
    
    elapsed = time.time() - start_time
    
    if stopped_by_client:
        logger.info(f"[ACCUMULATE] ** STOPPED by client | tokens_sent={total_tokens_from_ollama} | flushes={flush_count} | total_chars={len(accumulated)} | elapsed={elapsed:.2f}s")
        yield f"data: {json.dumps({'type': 'stopped', 'content': '', 'accumulated': accumulated})}\n\n"
    else:
        logger.info(f"[ACCUMULATE] ** DONE | total_tokens={total_tokens_from_ollama} | flushes={flush_count} | total_chars={len(accumulated)} | elapsed={elapsed:.2f}s")
        yield f"data: {json.dumps({'type': 'done', 'content': '', 'accumulated': accumulated})}\n\n"
    
    logger.info(f"[ACCUMULATE] ----------------------------------------------")


# --- SSE Streaming Strategy 2: Direct Yield (Real-time) ---
async def stream_direct(request: Request, model: str, messages: list):
    """
    Strategy 2: Yields each token immediately as it arrives from Ollama.
    Checks request.is_disconnected() to detect client stop.
    """
    accumulated = ""
    total_tokens = 0
    start_time = time.time()
    stopped_by_client = False
    
    user_msg = messages[-1]["content"][:50] + ("..." if len(messages[-1]["content"]) > 50 else "")
    logger.info(f"[DIRECT] ----------------------------------------------")
    logger.info(f"[DIRECT] Starting stream | model={model}")
    logger.info(f"[DIRECT] User message: \"{user_msg}\"")
    
    try:
        logger.info(f"[DIRECT] Calling ollama.chat(model={model}, stream=True)...")
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        logger.info(f"[DIRECT] Ollama stream opened successfully")
    except Exception as e:
        logger.error(f"[DIRECT] FAILED to open Ollama stream: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Ollama error: {str(e)}'})}\n\n"
        return
    
    for chunk_idx, chunk in enumerate(stream):
        # Check if client disconnected (clicked Stop)
        if await request.is_disconnected():
            logger.info(f"[DIRECT] Client disconnected - stopping stream")
            stopped_by_client = True
            # Explicitly close the Ollama stream generator to stop model generation
            # This tells Ollama's server to cancel the ongoing inference
            stream.close()
            logger.info(f"[DIRECT] Ollama stream generator closed - model stopped")
            break
        
        token = chunk.get("message", {}).get("content", "")
        
        if token:
            total_tokens += 1
            accumulated += token
            
            logger.debug(f"[DIRECT] Token #{total_tokens}: repr={repr(token)} | accumulated_len={len(accumulated)}")
            
            # Log every 10th token at INFO level for visibility
            if total_tokens % 10 == 0:
                logger.info(f"[DIRECT] ... token #{total_tokens} | accumulated_len={len(accumulated)}")
            
            yield f"data: {json.dumps({'type': 'chunk', 'content': token, 'accumulated': accumulated})}\n\n"
            await asyncio.sleep(0)  # Yield control to event loop
    
    elapsed = time.time() - start_time
    
    if stopped_by_client:
        logger.info(f"[DIRECT] ** STOPPED by client | tokens_sent={total_tokens} | total_chars={len(accumulated)} | elapsed={elapsed:.2f}s")
        yield f"data: {json.dumps({'type': 'stopped', 'content': '', 'accumulated': accumulated})}\n\n"
    else:
        logger.info(f"[DIRECT] ** DONE | total_tokens={total_tokens} | total_chars={len(accumulated)} | elapsed={elapsed:.2f}s")
        yield f"data: {json.dumps({'type': 'done', 'content': '', 'accumulated': accumulated})}\n\n"
    
    logger.info(f"[DIRECT] ----------------------------------------------")


# --- API Endpoints ---

@app.get("/")
async def get_index():
    """Serve the chat frontend"""
    logger.info(f"[HTTP] GET / - Serving index.html")
    html_path = Path(__file__).parent / "static" / "index.html"
    content = html_path.read_text(encoding="utf-8")
    logger.info(f"[HTTP] GET / - index.html served ({len(content)} bytes)")
    return HTMLResponse(content=content)


@app.post("/chat/accumulate")
async def chat_accumulate(request: Request):
    """
    Endpoint using Strategy 1: Accumulate-then-flush SSE streaming.
    The LLM response is buffered and flushed in batches.
    Uses request.is_disconnected() to detect when client stops.
    """
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "llama3.1:8b")
    flush_every = body.get("flush_every_chars", 5)
    
    logger.info(f"[HTTP] POST /chat/accumulate | model={model} | flush_every={flush_every} | messages_count={len(messages)}")
    logger.info(f"[HTTP]   Last message: \"{messages[-1]['content'][:80] if messages else 'N/A'}\"")
    
    return StreamingResponse(
        stream_accumulate(request, model, messages, flush_every),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/chat/direct")
async def chat_direct(request: Request):
    """
    Endpoint using Strategy 2: Direct yield SSE streaming.
    Every token is pushed immediately to the client.
    Uses request.is_disconnected() to detect when client stops.
    """
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "llama3.1:8b")
    
    logger.info(f"[HTTP] POST /chat/direct | model={model} | messages_count={len(messages)}")
    logger.info(f"[HTTP]   Last message: \"{messages[-1]['content'][:80] if messages else 'N/A'}\"")
    
    return StreamingResponse(
        stream_direct(request, model, messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/models")
async def list_models():
    """List available Ollama models"""
    logger.info(f"[HTTP] GET /models - Listing available models")
    try:
        models = ollama.list()
        model_names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
        logger.info(f"[HTTP] GET /models - Found {len(model_names)} models: {model_names}")
        return {"models": model_names}
    except Exception as e:
        logger.error(f"[HTTP] GET /models - Error: {e}")
        return {"models": [], "error": str(e)}