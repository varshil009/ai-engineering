"""Async Groq client wrapper for chat completions with tool calling and model fallback.

On HTTP 429 (rate limit) errors, automatically falls back to the backup model
defined in llm_limits.json.

Also strips think blocks from Qwen model responses.
"""

import re
import sys
from typing import Any

from groq import AsyncGroq
from groq import (
    RateLimitError as GroqRateLimitError,
)

from tool_calling_concepts.config import settings
from tool_calling_concepts.services.limits_manager import LimitsManager

# Regex to strip think blocks (Qwen models wrap reasoning in think tags)
_THINK_PATTERN = re.compile(r"", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove think blocks from a string."""
    return _THINK_PATTERN.sub("", text).strip()


class GroqClient:
    """Async wrapper around the Groq SDK for chat completions with fallback."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._limits = LimitsManager()

    @property
    def active_model(self) -> str:
        """Get the currently active model from the limits manager."""
        return self._limits.active_model

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Send a chat completion request with optional tool calling.

        On HTTP 429 (rate limit), automatically falls back to the backup model
        and retries once.

        For Qwen models, strips think blocks from the response content
        and tool call arguments.

        Args:
            messages: The conversation messages (system, user, assistant, tool).
            tools: Optional list of tool definitions the model can call.
            tool_choice: How the model selects tools ("auto", "required", or a specific tool).
            temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
            max_tokens: Maximum tokens in the response.

        Returns:
            The full response dict from Groq.

        Raises:
            RuntimeError: If API rate/token limits have been exceeded or request fails.
        """
        # Check limits before making the request
        if not self._limits.check_limits():
            summary = self._limits.usage_summary
            raise RuntimeError(
                f"API limits exceeded for model '{summary['model']}'. "
                f"Requests today: {summary['requests_today']}/{summary['requests_limit']}. "
                f"Tokens today: {summary['tokens_today']}/{summary['tokens_limit']}."
            )

        model = self.active_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # Track whether we are using a Qwen model (needs think-tag stripping)
        is_qwen = "qwen" in model.lower()

        # Attempt the request (with one fallback retry on 429)
        for attempt in range(2):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break  # Success - exit retry loop
            except GroqRateLimitError as exc:
                if attempt == 0:
                    fallback_model = self._limits.switch_to_fallback()
                    print(
                        f"[WARN] Rate limited on '{model}'. "
                        f"Falling back to '{fallback_model}'.",
                        flush=True,
                    )
                    sys.stdout.flush()
                    kwargs["model"] = fallback_model
                    is_qwen = "qwen" in fallback_model.lower()
                    continue
                raise RuntimeError(
                    f"Groq API rate limited on both primary and fallback models: {exc}"
                ) from exc
            except Exception as exc:
                # Check if this is a 429 rate limit error (Groq SDK may raise it as generic)
                exc_str = str(exc)
                if attempt == 0 and ("429" in exc_str or "rate_limit" in exc_str.lower() or "Rate limit" in exc_str):
                    fallback_model = self._limits.switch_to_fallback()
                    print(
                        f"[WARN] Rate limited on '{model}'. "
                        f"Falling back to '{fallback_model}'.",
                        flush=True,
                    )
                    sys.stdout.flush()
                    kwargs["model"] = fallback_model
                    is_qwen = "qwen" in fallback_model.lower()
                    continue
                raise RuntimeError(
                    f"Groq API request failed on model '{model}': {exc}"
                ) from exc
        else:
            raise RuntimeError(
                f"Groq API request failed after retrying fallback model."
            )

        try:
            result = response.model_dump(mode="json")
            # Strip 'annotations' field from assistant messages in the result,
            # as Groq's API rejects this field when re-sent in subsequent requests.
            try:
                choice = result["choices"][0]
                message = choice.get("message", {})
                if "annotations" in message:
                    del message["annotations"]
            except (KeyError, IndexError, TypeError):
                pass  # Malformed response — proceed without stripping
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse Groq API response: {exc}"
            ) from exc

        # Strip think blocks if using Qwen model
        if is_qwen:
            try:
                choice = result["choices"][0]
                message = choice.get("message", {})

                # Clean content
                content = message.get("content", "")
                if content:
                    message["content"] = _strip_think_tags(content)

                # Clean tool call arguments
                tool_calls = message.get("tool_calls", [])
                for tc in tool_calls:
                    fn_info = tc.get("function", {})
                    arguments = fn_info.get("arguments", "")
                    if arguments:
                        fn_info["arguments"] = _strip_think_tags(arguments)

            except (KeyError, IndexError, TypeError):
                pass  # Malformed response - return as-is

        # Record usage from the response
        try:
            usage = result.get("usage")
            self._limits.record_request(usage)
        except Exception as exc:
            print(f"Warning: Failed to record API usage: {exc}", flush=True)

        # Enforce 2-second delay between requests
        try:
            await self._limits.delay()
        except Exception as exc:
            print(f"Warning: Rate limiter delay failed: {exc}", flush=True)

        return result