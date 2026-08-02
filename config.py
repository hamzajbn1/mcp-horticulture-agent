from __future__ import annotations

import  logging 
from functools import lru_cache
from typing  import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGGER = logging.getLogger(__name__)

# Restrict log levels to standard safe values
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class Settings(BaseSettings):
    """Validated runtime configuration for the horticulture agent stack."""

    # Pydantic configuration: link to .env and lock the settings
    model_config = SettingsConfigDict(
        env_prefix="HORTI",
        env_file=".env",
        env_file_encoding= "utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Local LLM (Ollama) -------------------------------------------------
    ollama_model: str = Field(
        default= "qwen2.5",
        description = "Tool-calling capable Ollama model tag (e.g. 'qwen2.5', 'llama3.1').",
    )
    ollama_base_url: str =Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama daemon.",
    )
    ollama_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2, 
        description="Sampling temperature. 0.0 recommended for deterministic tool routing.",
    )
    ollama_num_ctx: int = Field(
        default=8192,
        ge=2048,
        description="Context window size requested from Ollama.",
    )

    # --- MCP server process -------------------------------------------------
    mcp_server_command: str = Field(
        default="python",
        description="Executable used to launch the MCP tool server over stdio.",
    )
    mcp_server_script: str = Field(
        default="mcp_server.py",
        description="Path to the FastMCP server entry point.",
    )

    # --- Agent behaviour ----------------------------------------------------
    recursion_limit: int = Field(
        default=15,
        ge=2,
        le=100,
        description="LangGraph recursion limit (hard stop for runaway tool loops).",
    )

    log_level: LogLevel = Field(default="INFO")

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_base_url(cls, value:str) -> str:
        if not value.startswith(("http://", "https://")):
            msg = "HORTI_OLLAMA_BASE_URL must be an http(s) URL"
            raise ValueError(msg)
        return value.rstrip("/")

# Cache the settings in RAM so the .env file is only read once
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    _LOGGER.debug("Settings loaded (model=%s)", settings.ollama_model)
    return settings

def configure_logging(level: LogLevel | None = None):
    """Set up standard terminal logging output."""
    resolved = level or get_settings().log_level
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )