# NICE NG12 Clinical RAG — Day 1 Retrieval Core

This repository contains a production-oriented ingestion and hybrid retrieval subsystem for **NICE NG12: _Suspected cancer: recognition and referral_**, pinned to the guideline last updated **15 April 2026**.[1] The current Day 1 snapshot covers four clinically diverse site groups: lung and pleural cancers (1.1), upper gastrointestinal cancers (1.2), lower gastrointestinal cancers (1.3), and urological cancers (1.6).

> **Safety boundary:** This is an educational hackathon prototype for retrieving guideline text. It does not diagnose cancer, replace clinical judgement, or constitute medical advice. The source guideline and local referral pathways remain authoritative.

## What Is Implemented

The parser validates the source PDF by SHA-256, title, author, update marker, and physical page count before extracting content. It then creates exactly one chunk per numbered recommendation, preserving physical PDF page provenance and rich clinical metadata. The current corpus contains **33 recommendation chunks** and **6 clearly labelled synthetic near-miss chunks** used only to test retrieval robustness.

| Layer | Implementation |
| --- | --- |
| Source control | Pinned 101-page April 2026 PDF with checksum validation |
| Parsing | Page-aware `pypdf` extraction with repeated header/footer removal |
| Chunking | One numbered recommendation per chunk; no fixed-size splitting |
| Metadata | Cancer site, recommendation ID, page(s), section/subsection, action, age condition, symptom condition, threshold, rule type, and revision history |
| Embeddings | OpenAI `text-embedding-3-small` when available; automatic local BGE fallback |
| Vector index | Normalised embeddings in FAISS `IndexFlatIP` |
| Lexical index | BM25 over clinically normalised tokens |
| Hybrid retrieval | Weighted reciprocal-rank fusion, exact recommendation/number boosts, threshold-conflict penalties, and cross-encoder reranking |
| Observability | Per-result vector, BM25, fusion, feature, reranker, and final scores logged as JSONL |

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/ng12_rag/ingestion.py` | Source validation and page-aware recommendation extraction |
| `src/ng12_rag/chunking.py` | Recommendation chunks, metadata, and synthetic negatives |
| `src/ng12_rag/embeddings.py` | OpenAI and local BGE embedding providers |
| `src/ng12_rag/indexing.py` | Persistent FAISS, BM25, metadata, and manifests |
| `src/ng12_rag/retrieval.py` | Hybrid fusion, reranking, filtering, and score logs |
| `src/ng12_rag/cli.py` | `ingest`, `index`, `build`, `search`, and `inspect` commands |
| `config/config.yaml` | Source pins, scope, models, top-K values, and thresholds |
| `tests/` | Parser, metadata, index-alignment, boundary, and negative-retrieval tests |

## Quick Start

Use Python 3.11 or newer. The first local-model build downloads the configured BGE embedding model and cross-encoder; subsequent runs use the local model cache.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Build every artifact from the pinned PDF:

```bash
ng12-rag build --embedding-provider local
```

If an OpenAI-compatible embeddings endpoint is available, use:

```bash
export OPENAI_API_KEY="..."
ng12-rag build --embedding-provider openai --embedding-model text-embedding-3-small
```

Run a hybrid retrieval query:

```bash
ng12-rag search "FIT result exactly 10 micrograms haemoglobin per gram faeces" --top-k 5
```

Inspect index provenance and dimensions:

```bash
ng12-rag inspect
```

Run the automated checks:

```bash
pytest -q
```

## Output Artifacts

`data/processed/chunks.jsonl` contains the 33 authoritative recommendation chunks. `data/processed/negative_chunks.jsonl` contains six synthetic near-miss records with explicit synthetic provenance. `data/index/vectors.faiss`, `metadata.jsonl`, `bm25_corpus.json`, and `index_manifest.json` form an aligned index set; the manifest includes the ordered corpus fingerprint and embedding model details.

Synthetic negatives may be indexed to stress-test candidate generation, but they receive a strong reranking penalty and are excluded from normal search results. They are never eligible as answer context.

## Scope Rationale

The four-site scope provides varied rule shapes within a two-day build: symptom combinations and smoking history in lung guidance, multi-branch upper-GI criteria, FIT thresholds and safety-netting in colorectal guidance, and age/laboratory branches in urological guidance. This diversity exercises exact keyword retrieval, semantic retrieval, numeric boundary handling, and recommendation-level provenance without pretending to cover all of NG12.

## Source and Rights

The source file is `data/raw/ng12.pdf`, with SHA-256 `140ecbe21a689a483f76fc5d05a954d759d4fab75773692df7b883124b691a27`. It remains subject to the copyright and usage notice printed in the document and the NICE website terms. This project is not affiliated with or endorsed by NICE.

## References

[1]: https://www.nice.org.uk/guidance/ng12 "NICE NG12: Suspected cancer — recognition and referral"
