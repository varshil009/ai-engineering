"""Rate-limit and token-limit manager for Groq API.

Reads/writes ``llm_limits.json`` to track:
    - Requests per day (RPD)
    - Tokens per day (TPD)
    - Whether more requests are allowed

Also enforces a 2-second delay between requests.
"""

import asyncio
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

_LIMITS_FILE = Path(__file__).resolve().parent.parent / "llm_limits.json"


class LimitsManager:
    """Manages API rate limits and token budgets."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = self._load()
        self._last_request_time: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────

    def check_limits(self) -> bool:
        """Check whether we can make another request.

        Returns:
            True if allowed, False if limits exceeded.
        """
        self._reset_if_new_day()

        usage = self._data["usage"]
        limits = self._data["limits"]

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
        self._reset_if_new_day()

        usage = self._data["usage"]
        usage["rpd_count"] += 1

        if usage_dict and "total_tokens" in usage_dict:
            usage["tpd_count"] += usage_dict["total_tokens"]

        self._save()

    async def delay(self) -> None:
        """Sleep for 2 seconds to respect rate limits."""
        await asyncio.sleep(2)

    @property
    def more_requests(self) -> bool:
        """Whether the system can still make API requests."""
        self._reset_if_new_day()
        return self._data["usage"]["more_requests"]

    @property
    def usage_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of current usage."""
        self._reset_if_new_day()
        usage = self._data["usage"]
        limits = self._data["limits"]
        return {
            "date": usage["date"],
            "requests_today": usage["rpd_count"],
            "requests_limit": limits["rpd"],
            "tokens_today": usage["tpd_count"],
            "tokens_limit": limits["tpd"],
            "more_requests_allowed": usage["more_requests"],
        }

    # ── Internal helpers ────────────────────────────────────────────────

    def _reset_if_new_day(self) -> None:
        """Reset counters if the date has changed."""
        today = date.today().isoformat()
        usage = self._data["usage"]
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
        return {
            "model_name": "llama-3.3-70b-versatile",
            "limits": {
                "tpm": 12000,
                "tpd": 500000,
                "rpm": 30,
                "rpd": 1000,
            },
            "usage": {
                "date": "",
                "rpd_count": 0,
                "tpd_count": 0,
                "more_requests": True,
            },
        }