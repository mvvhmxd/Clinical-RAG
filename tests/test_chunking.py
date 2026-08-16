from collections import Counter

from ng12_rag.chunking import create_chunks, create_negative_chunks
from ng12_rag.ingestion import extract_recommendations
from ng12_rag.models import ActionType, RuleType


def _corpus(settings):
    records, _ = extract_recommendations(settings.path("source.pdf_path"), settings)
    chunks = create_chunks(records, settings)
    return chunks, create_negative_chunks(chunks, settings)


def test_one_numbered_recommendation_is_one_chunk(settings):
    chunks, _ = _corpus(settings)
    assert len(chunks) == 33
    ids = [chunk.metadata.recommendation_id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert all(chunk.chunk_id.endswith(chunk.metadata.recommendation_id) for chunk in chunks)
    assert all(chunk.text.startswith(chunk.metadata.recommendation_id) for chunk in chunks)


def test_metadata_captures_clinical_conditions_and_provenance(settings):
    chunks, _ = _corpus(settings)
    by_id = {chunk.metadata.recommendation_id: chunk for chunk in chunks}

    lung = by_id["1.1.2"].metadata
    assert lung.cancer_site == "lung"
    assert lung.action_type == ActionType.OFFER_INVESTIGATION
    assert lung.age_condition == "aged 40 and over"
    assert lung.rule_type == RuleType.MULTI_BRANCH
    assert lung.page_number == 9

    fit = by_id["1.3.2"].metadata
    assert fit.action_type == ActionType.REFER
    assert "at least 10 micrograms" in (fit.lab_threshold or "")
    assert fit.rule_type == RuleType.THRESHOLD_BASED
    assert fit.revision_year == 2023

    oesophageal = by_id["1.2.1"].metadata
    assert oesophageal.revision_history == [2015, 2025]
    assert oesophageal.revision_year == 2025


def test_action_type_distribution_has_no_unclassified_chunks(settings):
    chunks, _ = _corpus(settings)
    counts = Counter(chunk.metadata.action_type for chunk in chunks)
    assert sum(counts.values()) == 33
    assert ActionType.CLINICAL_ASSESSMENT not in counts
    assert counts[ActionType.REFER] == 9
    assert counts[ActionType.CONSIDER_REFERRAL] == 8
    assert counts[ActionType.OFFER_INVESTIGATION] == 14


def test_synthetic_negatives_are_labelled_and_never_claim_source_status(settings):
    chunks, negatives = _corpus(settings)
    source_ids = {chunk.chunk_id for chunk in chunks}
    assert len(negatives) == 6
    for negative in negatives:
        assert negative.metadata.is_synthetic_negative
        assert negative.metadata.synthetic_source_id in source_ids
        assert negative.metadata.synthetic_mutation
        assert negative.text.startswith("[SYNTHETIC NEGATIVE — NOT NICE GUIDANCE]")
        assert "negative" in negative.chunk_id
