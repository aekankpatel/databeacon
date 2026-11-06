"""Runtime configuration, read from environment variables (prefix DATABEACON_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DATABEACON_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")).resolve())

    # LLM
    groq_model: str = field(default_factory=lambda: _env("MODEL", "openai/gpt-oss-120b"))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.1")))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "90")))

    # Agent budgets / stopping conditions
    max_iterations: int = field(default_factory=lambda: int(_env("MAX_ITERATIONS", "3")))
    max_tool_calls: int = field(default_factory=lambda: int(_env("MAX_TOOL_CALLS", "24")))
    max_steps_per_iteration: int = field(default_factory=lambda: int(_env("MAX_STEPS_PER_ITERATION", "8")))

    # SQL sandbox limits
    sql_timeout_s: float = field(default_factory=lambda: float(_env("SQL_TIMEOUT_S", "15")))
    sql_max_rows: int = field(default_factory=lambda: int(_env("SQL_MAX_ROWS", "5000")))
    llm_result_rows: int = field(default_factory=lambda: int(_env("LLM_RESULT_ROWS", "15")))

    # Uploads
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "200")))

    # MCP transport: "inprocess" (default) or "stdio" (servers run as subprocesses)
    mcp_transport: str = field(default_factory=lambda: _env("MCP_TRANSPORT", "inprocess"))

    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()
        )
    )

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def runs_db(self) -> Path:
        return self.data_dir / "runs.sqlite"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.datasets_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()
