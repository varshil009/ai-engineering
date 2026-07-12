"""Simple async Groq client for chat completions."""

from typing import Any, Optional

from groq import AsyncGroq


class GroqClient:
    """Simple async wrapper around the Groq SDK for chat completions."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._client = AsyncGroq(api_key=api_key)
        self.model = model

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Send a chat completion request and return the response dict."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = await self._client.chat.completions.create(**kwargs)
        return response.model_dump(mode="json")