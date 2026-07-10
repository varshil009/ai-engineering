"""Rate-limit and token-limit manager for Groq API with model fallback.

Reads/writes ``llm_limits.json`` to track:
    - Requests per day (RPD) and Tokens per day (TPD) per model
    - Active model and fallback model
    - Whether more requests are allowed

Also enforces a 2-second delay between requests.
"""

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

_LIMITS_FILE = Path(__file__).resolve().parent.parent / "llm_limits.json"

_MODELS = {
    "llama-3.3-70b-versatile": {"tpm": 12000, "tpd": 500000, "rpm": 30, "rpd": 1000},
    "qwen/qwen3-32b": {"tpm": 6000, "tpd": 500000, "rpm": 60, "rpd": 1000},
}


class LimitsManager:
    """Manages API rate limits and token budgets with multi-model support."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = self._load()
        self._last_request_time: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def active_model(self) -> str:
        """Get the currently active model name."""
        return self._data.get("active_model", "llama-3.3-70b-versatile")

    @active_model.setter
    def active_model(self, model_name: str) -> None:
        """Set the active model (e.g. on fallback)."""
        if model_name in self._data.get("models", {}):
            self._data["active_model"] = model_name
            self._save()

    @property
    def fallback_model(self) -> str:
        """Get the fallback model name."""
        return self._data.get("fallback_model", "qwen/qwen3-32b")

    def switch_to_fallback(self) -> str:
        """Switch active model to the fallback model.

        Returns:
            The fallback model name that is now active.
        """
        fb = self.fallback_model
        self.active_model = fb
        return fb

    def check_limits(self) -> bool:
        """Check whether we can make another request with the active model.

        Returns:
            True if allowed, False if limits exceeded.
        """
        active = self.active_model
        model_data = self._get_model_data(active)
        if model_data is None:
            return False

        self._reset_if_new_day(active)

        usage = model_data["usage"]
        limits = model_data["limits"]

        if not usage["more_requests"]:
            return False

        if usage["rpd_count"] >= limits["rpd"]:
            usage["more_requests"] = False
            self._save()
            return False

        if usage["tpd_count"] >= limits["tpd"]:
            usage["more_requests"] = False
            self._save()
            return False

        return True

    def record_request(self, usage_dict: Optional[dict[str, Any]] = None) -> None:
        """Record that a request was made and optionally track token usage.

        Args:
            usage_dict: The ``usage`` dict from the Groq response, e.g.
                ``{'total_tokens': 1727, ...}``. If ``None``, only the
                request count is incremented.
        """
        active = self.active_model
        model_data = self._get_model_data(active)
        if model_data is None:
            return

        self._reset_if_new_day(active)

        usage = model_data["usage"]
        usage["rpd_count"] += 1

        if usage_dict and "total_tokens" in usage_dict:
            usage["tpd_count"] += usage_dict["total_tokens"]

        self._save()

    async def delay(self) -> None:
        """Sleep for 2 seconds to respect rate limits."""
        await asyncio.sleep(2)

    @property
    def more_requests(self) -> bool:
        """Whether the system can still make API requests using active model."""
        active = self.active_model
        model_data = self._get_model_data(active)
        if model_data is None:
            return False
        self._reset_if_new_day(active)
        return model_data["usage"]["more_requests"]

    @property
    def usage_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of current usage for the active model."""
        active = self.active_model
        model_data = self._get_model_data(active)
        if model_data is None:
            return {"error": f"Model {active} not found"}
        self._reset_if_new_day(active)
        usage = model_data["usage"]
        limits = model_data["limits"]
        return {
            "model": active,
            "date": usage["date"],
            "requests_today": usage["rpd_count"],
            "requests_limit": limits["rpd"],
            "tokens_today": usage["tpd_count"],
            "tokens_limit": limits["tpd"],
            "more_requests_allowed": usage["more_requests"],
        }

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_model_data(self, model_name: str) -> Optional[dict[str, Any]]:
        """Get the model-specific data dict (limits + usage)."""
        models = self._data.get("models", {})
        return models.get(model_name)

    def _reset_if_new_day(self, model_name: str) -> None:
        """Reset counters if the date has changed for the given model."""
        model_data = self._get_model_data(model_name)
        if model_data is None:
            return
        today = date.today().isoformat()
        usage = model_data["usage"]
        if usage["date"] != today:
            usage["date"] = today
            usage["rpd_count"] = 0
            usage["tpd_count"] = 0
            usage["more_requests"] = True
            self._save()

    def _load(self) -> dict[str, Any]:
        """Load limits data from the JSON file."""
        if _LIMITS_FILE.exists():
            try:
                with open(_LIMITS_FILE, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_data()

    def _save(self) -> None:
        """Persist limits data to the JSON file."""
        try:
            with open(_LIMITS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass  # Best-effort — don't crash if file is unwritable

    @staticmethod
    def _default_data() -> dict[str, Any]:
        models_data: dict[str, Any] = {}
        for model_name, limits in _MODELS.items():
            models_data[model_name] = {
                "limits": limits,
                "usage": {
                    "date": "",
                    "rpd_count": 0,
                    "tpd_count": 0,
                    "more_requests": True,
                },
            }
        return {
            "active_model": "llama-3.3-70b-versatile",
            "fallback_model": "qwen/qwen3-32b",
            "models": models_data,
        }
