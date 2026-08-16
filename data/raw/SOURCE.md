# Pinned Source Document

This project ingests **NICE guideline NG12, _Suspected cancer: recognition and referral_**, published 23 June 2015 and **last updated 15 April 2026**. The supplied PDF reports 101 physical PDF pages and carries NICE publication metadata.

| Field | Value |
| --- | --- |
| Canonical guideline | <https://www.nice.org.uk/guidance/ng12> |
| Local file | `data/raw/ng12.pdf` |
| Guideline version | `2026-04-15` |
| PDF pages | `101` |
| SHA-256 | `140ecbe21a689a483f76fc5d05a954d759d4fab75773692df7b883124b691a27` |
| Publisher | National Institute for Health and Care Excellence (NICE) |

The ingestion command validates the title, page count, update text, and SHA-256 before producing any retrievable corpus. Strict validation is intentional: a silently changed guideline must never reuse an index built from a different clinical version.

The Day 1 corpus is deliberately restricted to sections **1.1, 1.2, 1.3, 1.6, and 1.13–1.16**. The project preserves the source wording and physical PDF page numbers. Synthetic negative chunks are clearly marked, excluded from answer context, and must never be represented as NICE recommendations.

The source PDF remains subject to the copyright and usage notice printed in the document and to the [NICE website terms](https://www.nice.org.uk/terms-and-conditions#notice-of-rights). This project is an educational hackathon prototype and is not affiliated with or endorsed by NICE.
