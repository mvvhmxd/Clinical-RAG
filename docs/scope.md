# Locked Corpus Scope

This file is the authoritative answer to "why these cancer sites?". The query pipeline must
refuse any question about a site not listed as in scope, and the same lock applies identically
to both corpus tiers — a site excluded from Tier 1 is also excluded from Tier 2.

> This is an educational engineering demonstration, not clinically validated decision support.
> Scope here is an engineering decision about what this build can defend, not a clinical
> judgement about which cancers matter.

## Why a lock is needed

A prior extraction of this corpus produced 33 recommendations spanning 14 cancer sites. That
was a side effect of selecting whole NG12 sections (1.1, 1.2, 1.3, 1.6) rather than a
deliberate decision — each of those sections contains subsections for several distinct sites.
Shipping all 14 would mean claiming coverage the team never reviewed. This file narrows that
to an explicit list and records the reason for every exclusion.

## In scope (7 sites, 20 recommendations)

| Cancer site | Section | Recommendation IDs | Why included |
| --- | --- | --- | --- |
| `lung` | 1.1 | 1.1.1, 1.1.2, 1.1.3 | Multi-branch symptom combinations plus smoking history; exercises count-based branching |
| `oesophageal` | 1.2 | 1.2.1, 1.2.2, 1.2.3 | Mixed multi-branch and clean inline age thresholds |
| `pancreatic` | 1.2 | 1.2.4, 1.2.5 | 1.2.4 is the age-AND-symptom case the condition-evaluation work is built around |
| `stomach` | 1.2 | 1.2.6, 1.2.7, 1.2.8, 1.2.9 | Four rule shapes across one site, including a threshold-based investigation rule |
| `colorectal` | 1.3 | 1.3.1, 1.3.2, 1.3.3, 1.3.4, 1.3.5 | The richest structure in the corpus: a multi-branch FIT/age/symptom rule, an exact lab threshold, and safety-netting |
| `renal` | 1.6 | 1.6.6 | Clean age-AND-symptom multi-branch rule sharing haematuria wording with bladder — a deliberate near-miss pair for retrieval testing |
| `bladder` | 1.6 | 1.6.4, 1.6.5 | Two age bands in one rule plus a lab condition; overlaps renal wording |

Selection favours **structural diversity of rules** over clinical breadth. Between these seven
sites the corpus contains single-condition rules, multi-branch OR rules, exact numeric lab
thresholds, inclusive age boundaries, and process/safety-netting recommendations — which is
what the retrieval and condition-evaluation layers need to be tested against. The renal and
bladder pair is retained specifically because both hinge on haematuria and similar age bands,
making them a genuine near-miss test rather than a synthetic one.

## Explicitly excluded (7 sites, 13 recommendations)

| Cancer site | Recommendation IDs | Reason for exclusion |
| --- | --- | --- |
| `mesothelioma` | 1.1.4, 1.1.5, 1.1.6 | Criteria hinge on asbestos-exposure history, an exposure-based condition type this build's condition evaluation does not model |
| `prostate` | 1.6.1, 1.6.2, 1.6.3 | PSA criteria require an age-banded table lookup, and external PCRMP guidance risks conflicting with NG12 |
| `testicular` | 1.6.7, 1.6.8 | Adds a site boundary without adding a rule shape not already covered |
| `penile` | 1.6.9, 1.6.10 | Adds a site boundary without adding a rule shape not already covered |
| `anal` | 1.3.6 | Single recommendation, duplicates branching shapes already covered by colorectal |
| `gall_bladder` | 1.2.10 | Incidental-imaging finding rather than a patient-presentation threshold rule |
| `liver` | 1.2.11 | Incidental-imaging finding rather than a patient-presentation threshold rule |

Excluded recommendations are preserved in `data/excluded/` for future extension. They must not
be reachable by the query pipeline, and a query naming an excluded site must be refused with a
message naming what *is* covered.

## Scope enforcement rules

1. A query naming an out-of-scope cancer site is refused outright.
2. A query mixing an in-scope and an out-of-scope site is refused **entirely**. Answering only
   the in-scope half produces something that looks like a complete answer and is a worse
   failure than a clean refusal.
3. The same site list gates Tier 2. A rationale chunk discussing prostate PSA evidence is
   excluded even though it lives in the full evidence guideline.
4. This list is the single source of truth. Guardrails read the locked sites from configuration
   derived from this file rather than maintaining a second hand-written list that can drift.

## Known corpus defects at time of locking

A regex sweep of the current Tier 1 corpus found extraction artifacts that must be cleaned
before this scope can be considered shippable. Recorded here so the lock is not mistaken for a
clean bill of health:

- **Page-footer bleed** in 10 of 33 recommendations (7 of them in scope: 1.1.2, 1.2.3, 1.2.5,
  1.2.9, 1.3.1, 1.3.3, 1.6.4). The fragment `conditions#notice-of-rights). 101` appears
  mid-sentence, in 1.3.1 splitting a bulleted list of referral criteria in two.
- **Next-subsection title bleed** in 7 of 33 recommendations (5 in scope: 1.1.3, 1.2.3, 1.2.9,
  1.6.5, 1.6.6). For example 1.6.6 ends `[2015] Testicular cancer`, appending the following
  subsection's heading to a renal recommendation.

Both defects affect `text` and `embedding_text`, so they degrade retrieval and would appear
inside a verbatim citation quote. Cleaning is tracked as Phase 1 work with a before/after diff
report.
