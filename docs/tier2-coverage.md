# Tier 2 Coverage and Limits

Tier 2 is the 382-page NG12 full evidence guideline (`data/raw/ng12_full.pdf`, June 2015). It
answers "why is this criterion set here", never "what are the criteria" — that is Tier 1's job.

Built by `python -m ng12_rag.tier2` after `python -m ng12_rag.ingestion_full`.

## What is in the retrievable corpus

| | Count |
| --- | --- |
| Raw chunks extracted (chapters 7, 8, 9, 12) | 93 |
| Active, linked, in-scope | 45 |
| Excluded | 48 |

Excluded chunks are kept in `data/excluded/tier2_excluded.jsonl` with a recorded reason and are
unreachable by the query pipeline.

| Exclusion reason | Count |
| --- | --- |
| Out of locked scope (prostate, mesothelioma, anal, liver, gall bladder, testicular, penile, small intestinal) | 27 |
| Ambiguous between recommendations within a site | 18 |
| Insufficient clinical content | 2 |
| Below link score floor | 1 |

## How chunks are linked to Tier 1

The 2015 full guideline never uses the short guideline's `1.x.y` numbering — a scan found the
pattern in exactly **1 of 103** raw chunks. Text matching is therefore not a viable linking
strategy.

Links are instead derived structurally:

1. The full guideline's cancer-site subsections (`7.1 Lung cancer`, `8.2 Pancreatic cancer`,
   `12.3 Renal cancer`, ...) correspond one-to-one with the short guideline's subsections. That
   fixes the candidate set to one site's recommendations.
2. Within a site, each candidate recommendation is scored by IDF-weighted term overlap against
   the chunk, so terms that distinguish sibling recommendations count for more than terms they
   share.
3. A link is accepted only if the best candidate clears a score floor **and** leads the
   runner-up by a margin. Otherwise the chunk is excluded.

Ambiguity produces an exclusion, never an arbitrary link. Rationale text attached to the wrong
criterion is a worse failure than rationale that is simply absent, because it looks correct.

## Known coverage gap: the 2015/2023 mismatch

**The colorectal FIT threshold has no rationale in this corpus, and the system must say so.**

Tier 1 recommendations 1.3.1–1.3.4 are adapted from NICE HealthTech guidance published in 2023.
The full evidence guideline predates that by eight years. A term scan of the entire 382-page
document finds:

| Term | Occurrences in Tier 2 |
| --- | --- |
| `10 micrograms` | 0 |
| `micrograms of haemoglobin` | 0 |
| `HM-JACKarc` | 0 |
| `OC-Sensor` | 0 |
| `faecal occult blood` | 34 |

So the question "why is the FIT threshold set at 10 µg/g?" **cannot** be answered from this
corpus. It is also an actively dangerous question, because the 2015 guideline discusses faecal
*occult blood* testing at length — superficially adjacent, clinically distinct, and easy for a
generation step to conflate into a confident, wrong rationale.

This question is therefore a required negative test case, not a demo query. The correct
behaviour is an explicit "the scoped evidence corpus does not contain a rationale for this
threshold" response.

Rationale queries that **are** genuinely supported, and are the right demo material:

| Term | Occurrences | Supports |
| --- | --- | --- |
| `haematuria` | 178 | renal 1.6.6, bladder 1.6.4/1.6.5 |
| `jaundice` | 33 | pancreatic 1.2.4 |
| `haemoptysis` | present | lung 1.1.1/1.1.2 |

## Other limits

- Only chapters 7, 8, 9 and 12 are extracted. General methodology, committee process, and
  health-economics chapters are not ingested, matching the rule that a chunk with no clear
  linked recommendation has no grounding value.
- Link distribution is uneven. 1.3.5 attracts 11 chunks and 1.2.6 only 1, which reflects how
  much the 2015 evidence review wrote about each topic, not the relative importance of the
  recommendations.
- Chunk sizes are 400–800 tokens, packed on paragraph and heading boundaries. Unlike Tier 1,
  chunk boundaries are not one-per-recommendation, because the evidence guideline has no
  equivalent atomic unit.
