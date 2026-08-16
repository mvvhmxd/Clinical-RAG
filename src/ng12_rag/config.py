"""Configuration loading and validation for the NG12 RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class Settings:
    """Immutable project settings with convenience accessors."""

    data: dict[str, Any]
    project_root: Path
    config_path: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ConfigurationError(f"Missing mapping configuration section: {name}")
        return value

    def path(self, dotted_key: str) -> Path:
        value: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ConfigurationError(f"Missing path setting: {dotted_key}")
            value = value[part]
        if not isinstance(value, str):
            raise ConfigurationError(f"Path setting must be a string: {dotted_key}")
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    @property
    def generation_model(self) -> str:
        return os.getenv("NG12_GENERATION_MODEL", self.section("generation")["model"])

    @property
    def embedding_provider(self) -> str:
        return os.getenv("NG12_EMBEDDING_PROVIDER", self.section("embedding")["provider"])

    @property
    def embedding_model(self) -> str:
        embedding = self.section("embedding")
        provider = self.embedding_provider
        if provider == "openai":
            return os.getenv("NG12_EMBEDDING_MODEL", embedding["openai_model"])
        return os.getenv("NG12_EMBEDDING_MODEL", embedding["local_model"])


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing ``config/config.yaml``."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "config" / "config.yaml").is_file():
            return candidate
    raise ConfigurationError(
        "Could not locate project root containing config/config.yaml; "
        "run from the repository or pass an explicit config path."
    )


def _apply_environment_overrides(data: dict[str, Any]) -> None:
    """Apply non-secret runtime overrides without mutating the YAML file."""

    mapping: dict[str, tuple[str, str]] = {
        "NG12_PDF_PATH": ("source", "pdf_path"),
        "NG12_CHUNKS_PATH": ("source", "chunks_path"),
        "NG12_GENERATION_MODEL": ("generation", "model"),
        "NG12_EMBEDDING_PROVIDER": ("embedding", "provider"),
    }
    for environment_key, (section, key) in mapping.items():
        value = os.getenv(environment_key)
        if value:
            data.setdefault(section, {})[key] = value


def _validate_settings(data: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "source",
        "scope",
        "chunking",
        "metadata",
        "negative_chunks",
        "embedding",
        "vector_store",
        "retrieval",
        "reranker",
        "scope_guard",
        "generation",
        "evaluation",
        "logging",
    }
    missing = sorted(required_sections - data.keys())
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {', '.join(missing)}")

    scope = data["scope"]
    include_sections = scope.get("include_sections", {})
    expected_sections = {"1.1", "1.2", "1.3", "1.6"}
    if set(include_sections) != expected_sections:
        raise ConfigurationError(
            "scope.include_sections must exactly match the approved four-site scope"
        )

    expected_count = scope.get("expected_recommendation_count")
    calculated_count = sum(
        int(item["end"]) - int(item["start"]) + 1
        for item in scope.get("required_id_ranges", [])
    )
    if expected_count != calculated_count:
        raise ConfigurationError(
            "expected_recommendation_count does not match required_id_ranges "
            f"({expected_count!r} != {calculated_count})"
        )

    retrieval = data["retrieval"]
    if not 0 <= float(retrieval["vector_weight"]) <= 1:
        raise ConfigurationError("retrieval.vector_weight must be between 0 and 1")
    if not 0 <= float(retrieval["bm25_weight"]) <= 1:
        raise ConfigurationError("retrieval.bm25_weight must be between 0 and 1")
    if abs(float(retrieval["vector_weight"]) + float(retrieval["bm25_weight"]) - 1) > 1e-9:
        raise ConfigurationError("retrieval vector and BM25 weights must sum to 1")


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load YAML settings and return paths anchored to the project root."""

    if config_path is None:
        project_root = find_project_root()
        path = project_root / "config" / "config.yaml"
    else:
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise ConfigurationError(f"Configuration file does not exist: {path}")
        project_root = path.parent.parent

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a YAML mapping")

    _apply_environment_overrides(loaded)
    _validate_settings(loaded)
    return Settings(data=loaded, project_root=project_root, config_path=path)
