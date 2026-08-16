"""Command-line orchestration for the NG12 ingestion and retrieval subsystem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ng12_rag.chunking import read_chunks, run_chunking
from ng12_rag.config import Settings, load_settings
from ng12_rag.embeddings import create_embedding_provider
from ng12_rag.indexing import build_index
from ng12_rag.ingestion import run_ingestion
from ng12_rag.logging_utils import configure_logging
from ng12_rag.retrieval import HybridRetriever


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config)
    logging_config = settings.section("logging")
    configure_logging(
        level=args.log_level or str(logging_config["level"]),
        json_format=not args.plain_logs,
    )
    return settings


def command_ingest(args: argparse.Namespace) -> None:
    settings = _settings(args)
    records = run_ingestion(settings)
    chunks, negatives = run_chunking(settings, records)
    _emit(
        {
            "status": "ok",
            "raw_recommendations": len(records),
            "recommendation_chunks": len(chunks),
            "synthetic_negative_chunks": len(negatives),
            "chunks_path": str(settings.path("source.chunks_path")),
        }
    )


def _provider(settings: Settings, args: argparse.Namespace):
    provider_override = getattr(args, "embedding_provider", None)
    model_override = getattr(args, "embedding_model", None)
    if not provider_override and not model_override:
        return None
    return create_embedding_provider(
        settings,
        provider_override=provider_override,
        model_override=model_override,
    )


def command_index(args: argparse.Namespace) -> None:
    settings = _settings(args)
    manifest = build_index(settings, embedding_provider=_provider(settings, args))
    _emit({"status": "ok", "index": manifest})


def command_build(args: argparse.Namespace) -> None:
    settings = _settings(args)
    records = run_ingestion(settings)
    chunks, negatives = run_chunking(settings, records)
    manifest = build_index(
        settings,
        chunks=chunks,
        negatives=negatives,
        embedding_provider=_provider(settings, args),
    )
    _emit(
        {
            "status": "ok",
            "raw_recommendations": len(records),
            "recommendation_chunks": len(chunks),
            "synthetic_negative_chunks": len(negatives),
            "index": manifest,
        }
    )


def command_search(args: argparse.Namespace) -> None:
    settings = _settings(args)
    retriever = HybridRetriever(settings)
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        include_synthetic_negatives=args.include_synthetic_negatives,
    )
    _emit(
        {
            "query": args.query,
            "result_count": len(results),
            "results": [result.model_dump(mode="json") for result in results],
        }
    )


def command_inspect(args: argparse.Namespace) -> None:
    settings = _settings(args)
    manifest_path = settings.path("vector_store.manifest_path")
    metadata_path = settings.path("vector_store.metadata_path")
    payload: dict[str, Any] = {
        "index_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "artifacts": {
            "faiss": str(settings.path("vector_store.index_path")),
            "metadata": str(metadata_path),
            "bm25": str(settings.path("vector_store.bm25_corpus_path")),
            "manifest": str(manifest_path),
        },
        "metadata_count": len(read_chunks(metadata_path)),
    }
    _emit(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ng12-rag",
        description=(
            "Build and query the four-site NICE NG12 recommendation retrieval index."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml; defaults to config/config.yaml under the repository.",
    )
    parser.add_argument("--log-level", default=None)
    parser.add_argument(
        "--plain-logs", action="store_true", help="Use human-readable instead of JSON logs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="Validate the PDF and create recommendation-level chunks."
    )
    ingest.set_defaults(handler=command_ingest)

    index = subparsers.add_parser(
        "index", help="Build FAISS and BM25 indexes from existing chunks."
    )
    index.add_argument(
        "--embedding-provider",
        choices=("auto", "openai", "local", "local_sentence_transformer"),
        default=None,
    )
    index.add_argument("--embedding-model", default=None)
    index.set_defaults(handler=command_index)

    build = subparsers.add_parser(
        "build", help="Run source validation, ingestion, chunking, and indexing."
    )
    build.add_argument(
        "--embedding-provider",
        choices=("auto", "openai", "local", "local_sentence_transformer"),
        default=None,
    )
    build.add_argument("--embedding-model", default=None)
    build.set_defaults(handler=command_build)

    search = subparsers.add_parser(
        "search", help="Run hybrid retrieval against the persisted index."
    )
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=None)
    search.add_argument("--include-synthetic-negatives", action="store_true")
    search.set_defaults(handler=command_search)

    inspect = subparsers.add_parser(
        "inspect", help="Print index metadata and artifact locations."
    )
    inspect.set_defaults(handler=command_inspect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
