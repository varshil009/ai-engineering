"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from typing import Final


# ──────────────────────────────────────────────
# Table schema metadata — used in system prompt
# ──────────────────────────────────────────────

TABLE_SCHEMAS: Final[dict[str, list[dict[str, str]]]] = {
    "BOLNEY": [
        {"column": "ID", "type": "bigint", "description": "Primary key"},
        {"column": "Date", "type": "timestamp with time zone", "description": "Timestamp of the record"},
        {"column": "SGT", "type": "text", "description": "Super Grid Transformer identifier"},
        {"column": "ActivePower_Avg", "type": "double precision", "description": "Average active power reading"},
    ],
    "SGT1": [
        {"column": "sgt", "type": "text", "description": "Super Grid Transformer identifier"},
        {"column": "prediction_index", "type": "bigint", "description": "Prediction sequence index"},
        {"column": "prediction_generated_at_utc", "type": "timestamp with time zone", "description": "When the prediction was generated"},
        {"column": "predicted_for_utc", "type": "timestamp with time zone", "description": "The time the prediction is for"},
        {"column": "input_window_start_utc", "type": "timestamp with time zone", "description": "Start of input window"},
        {"column": "input_window_end_utc", "type": "timestamp with time zone", "description": "End of input window"},
        {"column": "input_window_points", "type": "bigint", "description": "Number of data points in input window"},
        {"column": "input_window_minutes", "type": "bigint", "description": "Duration of input window in minutes"},
        {"column": "forecast_horizon_minutes", "type": "bigint", "description": "Forecast horizon in minutes"},
        {"column": "actual", "type": "double precision", "description": "Actual measured value"},
        {"column": "predicted", "type": "double precision", "description": "Predicted value"},
        {"column": "residual", "type": "double precision", "description": "Difference between actual and predicted"},
        {"column": "absolute_error", "type": "double precision", "description": "Absolute error"},
        {"column": "absolute_percentage_error", "type": "double precision", "description": "Absolute percentage error"},
        {"column": "source_file", "type": "text", "description": "Source file name"},
    ],
    "SGT2": [
        {"column": "sgt", "type": "text", "description": "Super Grid Transformer identifier"},
        {"column": "prediction_index", "type": "bigint", "description": "Prediction sequence index"},
        {"column": "prediction_generated_at_utc", "type": "timestamp with time zone", "description": "When the prediction was generated"},
        {"column": "predicted_for_utc", "type": "timestamp with time zone", "description": "The time the prediction is for"},
        {"column": "input_window_start_utc", "type": "timestamp with time zone", "description": "Start of input window"},
        {"column": "input_window_end_utc", "type": "timestamp with time zone", "description": "End of input window"},
        {"column": "input_window_points", "type": "bigint", "description": "Number of data points in input window"},
        {"column": "input_window_minutes", "type": "bigint", "description": "Duration of input window in minutes"},
        {"column": "forecast_horizon_minutes", "type": "bigint", "description": "Forecast horizon in minutes"},
        {"column": "actual", "type": "double precision", "description": "Actual measured value"},
        {"column": "predicted", "type": "double precision", "description": "Predicted value"},
        {"column": "residual", "type": "double precision", "description": "Difference between actual and predicted"},
        {"column": "absolute_error", "type": "double precision", "description": "Absolute error"},
        {"column": "absolute_percentage_error", "type": "double precision", "description": "Absolute percentage error"},
        {"column": "source_file", "type": "text", "description": "Source file name"},
    ],
    "SGT3": [
        {"column": "sgt", "type": "text", "description": "Super Grid Transformer identifier"},
        {"column": "prediction_index", "type": "bigint", "description": "Prediction sequence index"},
        {"column": "prediction_generated_at_utc", "type": "timestamp with time zone", "description": "When the prediction was generated"},
        {"column": "predicted_for_utc", "type": "timestamp with time zone", "description": "The time the prediction is for"},
        {"column": "input_window_start_utc", "type": "timestamp with time zone", "description": "Start of input window"},
        {"column": "input_window_end_utc", "type": "timestamp with time zone", "description": "End of input window"},
        {"column": "input_window_points", "type": "bigint", "description": "Number of data points in input window"},
        {"column": "input_window_minutes", "type": "bigint", "description": "Duration of input window in minutes"},
        {"column": "forecast_horizon_minutes", "type": "bigint", "description": "Forecast horizon in minutes"},
        {"column": "actual", "type": "double precision", "description": "Actual measured value"},
        {"column": "predicted", "type": "double precision", "description": "Predicted value"},
        {"column": "residual", "type": "double precision", "description": "Difference between actual and predicted"},
        {"column": "absolute_error", "type": "double precision", "description": "Absolute error"},
        {"column": "absolute_percentage_error", "type": "double precision", "description": "Absolute percentage error"},
        {"column": "source_file", "type": "text", "description": "Source file name"},
    ],
    "SGT4": [
        {"column": "sgt", "type": "text", "description": "Super Grid Transformer identifier"},
        {"column": "prediction_index", "type": "bigint", "description": "Prediction sequence index"},
        {"column": "prediction_generated_at_utc", "type": "timestamp with time zone", "description": "When the prediction was generated"},
        {"column": "predicted_for_utc", "type": "timestamp with time zone", "description": "The time the prediction is for"},
        {"column": "input_window_start_utc", "type": "timestamp with time zone", "description": "Start of input window"},
        {"column": "input_window_end_utc", "type": "timestamp with time zone", "description": "End of input window"},
        {"column": "input_window_points", "type": "bigint", "description": "Number of data points in input window"},
        {"column": "input_window_minutes", "type": "bigint", "description": "Duration of input window in minutes"},
        {"column": "forecast_horizon_minutes", "type": "bigint", "description": "Forecast horizon in minutes"},
        {"column": "actual", "type": "double precision", "description": "Actual measured value"},
        {"column": "predicted", "type": "double precision", "description": "Predicted value"},
        {"column": "residual", "type": "double precision", "description": "Difference between actual and predicted"},
        {"column": "absolute_error", "type": "double precision", "description": "Absolute error"},
        {"column": "absolute_percentage_error", "type": "double precision", "description": "Absolute percentage error"},
        {"column": "source_file", "type": "text", "description": "Source file name"},
    ],
}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment."""

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables (already loaded via dotenv or shell)."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            supabase_url=os.getenv("SUPABASE_PROJECT_URL", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_API", ""),
        )

    def validate(self) -> None:
        """Raise ValueError if any required setting is missing."""
        missing: list[str] = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.supabase_url:
            missing.append("SUPABASE_PROJECT_URL")
        if not self.supabase_service_role_key:
            missing.append("SUPABASE_SERVICE_ROLE_API")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Ensure they are set in the .env file or shell environment."
            )


# Module-level singleton — loaded once on import
settings: Settings = Settings.from_env()