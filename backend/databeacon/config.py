"""Runtime configuration, read from environment variables (prefix DATABEACON_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DATABEACON_{name}", default)


def _public(public_default: str, local_default: str) -> str:
    """Default that depends on DATABEACON_PUBLIC_MODE (stricter for public deployments)."""
    return public_default if _env("PUBLIC_MODE", "0").lower() in {"1", "true", "yes"} else local_default


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

    # Public deployment: per-visitor isolation, rate limits, retention, no internals in responses.
    # Every limit below can be overridden; PUBLIC_MODE only changes the defaults.
    public_mode: bool = field(default_factory=lambda: _env("PUBLIC_MODE", "0").lower() in {"1", "true", "yes"})

    # Uploads and dataset size
    max_upload_mb: float = field(default_factory=lambda: float(_env("MAX_UPLOAD_MB", _public("4", "200"))))
    max_rows: int = field(default_factory=lambda: int(_env("MAX_ROWS", _public("250000", "5000000"))))
    max_columns: int = field(default_factory=lambda: int(_env("MAX_COLUMNS", _public("100", "500"))))
    max_datasets_per_session: int = field(default_factory=lambda: int(_env("MAX_DATASETS_PER_SESSION", _public("3", "100"))))
    duckdb_memory_limit: str = field(default_factory=lambda: _env("DUCKDB_MEMORY_LIMIT", _public("512MB", "1GB")))
    # hours after which datasets, runs and reports are deleted (0 = keep)
    data_ttl_hours: float = field(default_factory=lambda: float(_env("DATA_TTL_HOURS", _public("24", "0"))))

    # Rate limits (per client IP; 0 disables)
    runs_per_hour: int = field(default_factory=lambda: int(_env("RUNS_PER_HOUR", _public("3", "0"))))
    runs_per_day: int = field(default_factory=lambda: int(_env("RUNS_PER_DAY", _public("8", "0"))))
    uploads_per_hour: int = field(default_factory=lambda: int(_env("UPLOADS_PER_HOUR", _public("20", "0"))))
    requests_per_minute: int = field(default_factory=lambda: int(_env("REQUESTS_PER_MINUTE", _public("120", "0"))))
    # trust X-Forwarded-For (only behind a proxy you control, e.g. Vercel / a reverse proxy)
    trust_proxy: bool = field(default_factory=lambda: _env("TRUST_PROXY", "0").lower() in {"1", "true", "yes"})
    # optional shared code required to start live (LLM-spending) analyses
    demo_access_code: str = field(default_factory=lambda: _env("DEMO_ACCESS_CODE", ""))
    expose_docs: bool = field(default_factory=lambda: _env("EXPOSE_DOCS", _public("0", "1")) == "1")

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
