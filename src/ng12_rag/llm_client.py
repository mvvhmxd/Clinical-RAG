"""Gemini backend exposing the minimal OpenAI chat-completions surface.

``GroundedGenerator`` depends on exactly one client capability: a structured
``chat.completions.create`` call that returns JSON conforming to a Pydantic-derived schema.
This module supplies that surface for Google's Generative Language API so the generation layer
does not need provider-specific branches, and so tests can keep injecting fake clients.

Two provider differences are handled here rather than leaking upstream:

* Gemini accepts a restricted OpenAPI-flavoured schema. Pydantic emits ``$defs``/``$ref`` for
  nested models and ``anyOf`` for optionals, both of which Gemini rejects, so schemas are
  inlined and rewritten by :func:`to_gemini_schema`.
* Gemini takes the system prompt as a separate ``systemInstruction`` rather than a message.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.7-flash"

# Retried because they are transient; every other status fails fast so a real
# configuration error is not mistaken for a blip and silently retried.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Validation keywords Pydantic emits that Gemini's schema dialect rejects. Dropping them is
# safe: the response is validated against the real Pydantic model after parsing, so these
# constraints are still enforced, just one layer later.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "title",
        "default",
        "additionalProperties",
        "$schema",
        "definitions",
        "discriminator",
        "examples",
        "const",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "pattern",
        "minItems",
        "maxItems",
    }
)

_SUPPORTED_SCHEMA_KEYS = frozenset(
    {"type", "description", "enum", "items", "properties", "required", "nullable"}
)


class GeminiError(RuntimeError):
    """Raised when the Gemini API cannot produce a usable structured response."""


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        raise GeminiError(f"Unsupported schema reference: {ref}")
    name = ref[len(prefix) :]
    if name not in defs:
        raise GeminiError(f"Schema reference {ref} has no matching definition")
    return defs[name]


def _convert(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    """Rewrite one JSON Schema node into Gemini's accepted subset."""

    if depth > 25:
        raise GeminiError("Schema nesting exceeded the supported depth")
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        return _convert(_resolve_ref(node["$ref"], defs), defs, depth + 1)

    # Pydantic renders `str | None` as anyOf[type, null]. Gemini expresses the same thing as
    # a nullable scalar, so collapse to the non-null branch and mark it nullable.
    if "anyOf" in node:
        branches = [b for b in node["anyOf"] if b.get("type") != "null"]
        nullable = len(branches) != len(node["anyOf"])
        if not branches:
            return {"type": "string", "nullable": True}
        converted = _convert(branches[0], defs, depth + 1)
        if nullable and isinstance(converted, dict):
            converted["nullable"] = True
        return converted

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS or key == "$defs":
            continue
        if key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _convert(v, defs, depth + 1) for k, v in value.items()}
        elif key == "items":
            out["items"] = _convert(value, defs, depth + 1)
        elif key in _SUPPORTED_SCHEMA_KEYS:
            out[key] = value

    if out.get("type") == "object" and "properties" not in out:
        # Gemini rejects an object with no declared properties.
        out["properties"] = {"value": {"type": "string"}}
    return out


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic ``model_json_schema()`` payload into a Gemini response schema."""

    defs = schema.get("$defs", {})
    converted = _convert(schema, defs)
    if not isinstance(converted, dict):
        raise GeminiError("Schema did not convert to an object")
    return converted


@dataclass(frozen=True)
class _Message:
    content: str
    refusal: None = None


@dataclass(frozen=True)
class _Choice:
    message: _Message
    finish_reason: str | None = None


@dataclass(frozen=True)
class _Completion:
    """OpenAI-shaped result so callers need no provider-specific unwrapping."""

    choices: list[_Choice]
    model: str
    usage: dict[str, Any] | None = None


class _Completions:
    def __init__(self, client: GeminiClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,
        **_ignored: Any,
    ) -> _Completion:
        system_prompt = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        ).strip()
        user_prompt = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") != "system"
        ).strip()
        if not user_prompt:
            raise GeminiError("No user content supplied to the model")

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        schema = _extract_schema(response_format)
        if schema is not None:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = schema

        data = self._client._post(f"models/{model}:generateContent", payload, timeout=timeout)
        return _Completion(
            choices=[_Choice(message=_Message(content=_extract_text(data)))],
            model=model,
            usage=data.get("usageMetadata"),
        )


def _extract_schema(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format:
        return None
    if response_format.get("type") != "json_schema":
        return None
    raw = response_format.get("json_schema", {}).get("schema")
    if not isinstance(raw, dict):
        return None
    return to_gemini_schema(raw)


def _extract_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback", {})
        blocked = feedback.get("blockReason")
        if blocked:
            raise GeminiError(f"Gemini blocked the prompt: {blocked}")
        raise GeminiError("Gemini returned no candidates")

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    parts = candidate.get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        # MAX_TOKENS with empty text is a silent truncation; surfacing it prevents the
        # generation layer from reporting a spurious "model returned no JSON".
        raise GeminiError(f"Gemini returned empty content (finishReason={finish})")
    return text


class _Chat:
    def __init__(self, client: GeminiClient) -> None:
        self.completions = _Completions(client)


class GeminiClient:
    """Minimal Gemini client exposing ``client.chat.completions.create``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        default_timeout: float = 60.0,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise GeminiError("GEMINI_API_KEY is not set")
        self._api_key = key
        self._base_url = (base_url or os.getenv("GEMINI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._max_retries = max(1, max_retries)
        self._default_timeout = default_timeout
        self.chat = _Chat(self)

    def _post(
        self, path: str, payload: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        url = f"{self._base_url}/{path}"
        body = json.dumps(payload).encode("utf-8")
        effective_timeout = timeout or self._default_timeout
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=effective_timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                last_error = GeminiError(f"Gemini HTTP {exc.code}: {detail}")
                if exc.code not in _RETRYABLE_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = GeminiError(f"Gemini request failed: {exc}")

            if attempt < self._max_retries - 1:
                backoff = (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "gemini_retry", extra={"attempt": attempt + 1, "sleep_seconds": backoff}
                )
                time.sleep(backoff)

        raise last_error or GeminiError("Gemini request failed")


def active_provider(provider: str | None = None) -> str:
    """Return the provider name that :func:`create_llm_client` would select."""

    selected = (provider or os.getenv("NG12_LLM_PROVIDER") or "").strip().lower()
    if selected == "openai":
        return "openai"
    if selected in {"gemini", "google"} or os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "openai"


def default_generation_model() -> str:
    """Return the default model for the active provider."""

    if active_provider() == "gemini":
        return os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    return "gpt-5-mini"


def create_llm_client(provider: str | None = None) -> Any:
    """Return a chat client for the configured provider.

    Defaults to Gemini when ``GEMINI_API_KEY`` is present, falling back to OpenAI so an
    existing OpenAI-configured environment keeps working unchanged.
    """

    selected = (provider or os.getenv("NG12_LLM_PROVIDER") or "").strip().lower()
    if selected == "openai":
        from openai import OpenAI

        return OpenAI()
    if selected in {"gemini", "google"} or os.getenv("GEMINI_API_KEY"):
        return GeminiClient()

    from openai import OpenAI

    return OpenAI()
