"""Async Groq client wrapper for chat completions with tool calling."""

from typing import Any

from groq import AsyncGroq

from tool_calling_concepts.config import settings


class GroqClient:
    """Async wrapper around the Groq SDK for chat completions."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model: str = settings.groq_model

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a chat completion request with optional tool calling.

        Args:
            messages: The conversation messages (system, user, assistant, tool).
            tools: Optional list of tool definitions the model can call.
            tool_choice: How the model selects tools ("auto", "required", or a specific tool).
            temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
            max_tokens: Maximum tokens in the response.

        Returns:
            The full response dict from Groq.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = await self._client.chat.completions.create(**kwargs)

        # Convert to a plain dict for easy serialisation
        return response.model_dump(mode="json")