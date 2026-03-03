"""Configuration management via Pydantic Settings + YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaConfig(BaseSettings):
    base_url: str = "http://ollama:11434"
    request_timeout_seconds: int = 120
    keep_alive: str = "5m"


class ModelsConfig(BaseSettings):
    planner: str = "deepseek-r1:14b"
    gatherer: str = "qwen2.5:7b"
    analyst: str = "qwen2.5:14b"
    critic: str = "deepseek-r1:7b"
    writer: str = "mistral-nemo:12b"
    embedding: str = "nomic-embed-text"
    eval_judge: str = "qwen2.5:14b"
    researcher: str = "qwen2.5:14b"
    fallbacks: dict[str, str] = Field(default_factory=lambda: {
        "planner": "qwen2.5:7b",
        "analyst": "qwen2.5:7b",
        "critic": "qwen2.5:7b",
        "writer": "qwen2.5:7b",
    })


class ChunkingConfig(BaseSettings):
    chunk_size: int = 1500
    chunk_overlap: int = 200
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


class RetrievalConfig(BaseSettings):
    vector_candidates: int = 20
    bm25_candidates: int = 20
    final_top_k: int = 5
    embedding_prefix_document: str = "search_document: "
    embedding_prefix_query: str = "search_query: "


class PipelineConfig(BaseSettings):
    max_critic_retries: int = 2
    quality_threshold_for_corpus: float = 0.6
    max_concurrent_pipelines: int = 1


class StorageConfig(BaseSettings):
    data_dir: str = "./data"
    vector_db_path: str = "./data/lancedb"
    metadata_db_path: str = "./data/metadata.db"
    checkpoints_db_path: str = "./data/checkpoints.db"
    briefings_dir: str = "./data/briefings"
    logs_dir: str = "./logs"


class WebSearchConfig(BaseSettings):
    mode: str = "auto"  # "auto" | "always" | "disabled"
    max_results: int = 3
    max_page_chars: int = 8000
    fetch_timeout_seconds: int = 15


class WebConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000


class McpConfig(BaseSettings):
    transport: str = "stdio"
    sse_host: str = "0.0.0.0"
    sse_port: int = 8001


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESEARCHFORGE_",
        env_nested_delimiter="__",
    )

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """Load settings from YAML file, with environment variable overrides."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        return Settings(**yaml_data)
    return Settings()


# Module-level singleton — import this wherever settings are needed.
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
