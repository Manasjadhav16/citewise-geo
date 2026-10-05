"""Stage 1 configuration, read from environment variables."""
import os
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

from ..geo_metrics import load_source_categories


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class Stage1Config:
    runs_per_query: int = 15  # N repeated observations per query
    query_count: int = 10  # queries selected when generating them
    evidence_token_budget: int = 20000  # total evidence tokens per query
    per_source_token_cap: int = 1000
    min_source_tokens: int = 200  # floor for a meaningful representation
    compression_batch_size: int = 5
    observation_concurrency: int = 4  # grounded calls in flight (the rate limiter still paces them)
    crawl_concurrency: int = 5
    max_page_chars: int = 40000  # page text passed to evidence extraction
    classification_batch_size: int = 25  # domains per categorisation call
    fetcher: str = "gemini_grounding"
    grounding_model: str = "gemini-2.5-flash"
    # serpapi_aio fetcher: fixed for the whole run
    serpapi_gl: str = "in"
    serpapi_hl: str = "en"
    serpapi_device: str = "desktop"
    # Observation-only runs: skip crawling cited pages and evidence extraction (saves LLM quota);
    # LLM categorisation still runs
    skip_evidence: bool = False
    source_categories: Tuple[str, ...] = load_source_categories("")

    @property
    def serpapi_settings(self) -> dict:
        return {"gl": self.serpapi_gl, "hl": self.serpapi_hl, "device": self.serpapi_device}

    @property
    def max_sources_per_query(self) -> int:
        """Most sources that fit the evidence budget at the per-source floor."""
        return max(1, self.evidence_token_budget // self.min_source_tokens)

    @classmethod
    def from_env(cls, overrides: Optional[Mapping[str, Any]] = None) -> "Stage1Config":
        config = cls(
            runs_per_query=_env_int("STAGE1_RUNS_PER_QUERY", cls.runs_per_query),
            query_count=_env_int("STAGE1_QUERY_COUNT", cls.query_count),
            evidence_token_budget=_env_int("STAGE1_EVIDENCE_TOKEN_BUDGET", cls.evidence_token_budget),
            per_source_token_cap=_env_int("STAGE1_PER_SOURCE_TOKEN_CAP", cls.per_source_token_cap),
            min_source_tokens=_env_int("STAGE1_MIN_SOURCE_TOKENS", cls.min_source_tokens),
            compression_batch_size=_env_int("STAGE1_COMPRESSION_BATCH_SIZE", cls.compression_batch_size),
            observation_concurrency=_env_int("STAGE1_OBSERVATION_CONCURRENCY", cls.observation_concurrency),
            crawl_concurrency=_env_int("STAGE1_CRAWL_CONCURRENCY", cls.crawl_concurrency),
            max_page_chars=_env_int("STAGE1_MAX_PAGE_CHARS", cls.max_page_chars),
            classification_batch_size=_env_int("STAGE1_CLASSIFICATION_BATCH_SIZE", cls.classification_batch_size),
            fetcher=os.getenv("STAGE1_FETCHER", cls.fetcher),
            grounding_model=os.getenv("STAGE1_GROUNDING_MODEL", cls.grounding_model),
            serpapi_gl=os.getenv("STAGE1_SERPAPI_GL", cls.serpapi_gl),
            serpapi_hl=os.getenv("STAGE1_SERPAPI_HL", cls.serpapi_hl),
            serpapi_device=os.getenv("STAGE1_SERPAPI_DEVICE", cls.serpapi_device),
            skip_evidence=_env_bool("STAGE1_SKIP_EVIDENCE", cls.skip_evidence),
            source_categories=load_source_categories(),
        )
        # Per-run overrides from the API request (e.g. a smaller N for a quick run)
        if overrides:
            config = replace(config, **{k: v for k, v in overrides.items() if v is not None})
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "runs_per_query",
            "query_count",
            "evidence_token_budget",
            "per_source_token_cap",
            "min_source_tokens",
            "compression_batch_size",
            "observation_concurrency",
            "crawl_concurrency",
            "max_page_chars",
            "classification_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"Stage 1 config: {name} must be >= 1")
        if self.fetcher not in ("gemini_grounding", "serpapi_aio"):
            raise ValueError(f"Stage 1 config: unknown fetcher {self.fetcher!r}")
        if self.serpapi_device not in ("desktop", "mobile", "tablet"):
            raise ValueError("Stage 1 config: serpapi_device must be desktop, mobile or tablet")
        if self.min_source_tokens > self.per_source_token_cap:
            raise ValueError("Stage 1 config: min_source_tokens cannot exceed per_source_token_cap")
