import re

import pytest

from ng12_rag.ingestion import (
    SUBSECTION_RANGES,
    expected_recommendation_ids,
    extract_recommendations,
    validate_source,
)

# One entry per extraction defect class. Swept across every record and both text fields,
# because the previous spot-check of four recommendation IDs passed while 1.6.6 still ended
# with the next subsection's heading and ten records still carried page-footer text.
PUBLICATION_ARTIFACTS: dict[str, re.Pattern[str]] = {
    "page_footer_url": re.compile(r"conditions#notice-of-rights", re.IGNORECASE),
    "notice_of_rights_prose": re.compile(r"Subject to Notice of rights", re.IGNORECASE),
    "copyright_line": re.compile(r"©\s*NICE", re.IGNORECASE),
    "page_x_of_y": re.compile(r"\bPage\s+\d+\s+of\b", re.IGNORECASE),
    "bare_page_total": re.compile(r"(?<![\d.])101\b"),
    "running_header": re.compile(r"Suspected cancer:\s*recognition and referral", re.IGNORECASE),
}


def test_source_is_exact_pinned_april_2026_pdf(settings):
    result = validate_source(settings.path("source.pdf_path"), settings)
    assert result.valid
    assert result.page_count == 101
    assert result.sha256 == settings.section("source")["expected_sha256"]
    assert "Suspected cancer" in result.title
    assert "NICE" in result.author


def test_extracts_exactly_the_four_scoped_sections(settings):
    records, _ = extract_recommendations(settings.path("source.pdf_path"), settings)
    assert len(records) == 33
    assert [record.recommendation_id for record in records] == expected_recommendation_ids(
        settings
    )
    assert {record.section_id for record in records} == {"1.1", "1.2", "1.3", "1.6"}


def test_recommendations_preserve_physical_pdf_pages(settings):
    records, _ = extract_recommendations(settings.path("source.pdf_path"), settings)
    by_id = {record.recommendation_id: record for record in records}
    assert by_id["1.1.1"].page_numbers == [9]
    assert by_id["1.2.3"].page_numbers == [11, 12]
    assert by_id["1.3.1"].page_numbers == [14, 15]
    assert by_id["1.6.4"].page_numbers == [22, 23]
    assert all(record.page_number == record.page_numbers[0] for record in records)


def test_section_and_subsection_headings_do_not_leak_into_chunks(settings):
    records, _ = extract_recommendations(settings.path("source.pdf_path"), settings)
    by_id = {record.recommendation_id: record for record in records}
    assert "Mesothelioma" not in by_id["1.1.3"].text
    assert "Pancreatic cancer" not in by_id["1.2.3"].text
    assert "Anal cancer" not in by_id["1.3.5"].text
    assert "Bladder cancer" not in by_id["1.6.3"].text
    assert by_id["1.6.10"].text.endswith("[2015]")


@pytest.mark.parametrize("artifact_name", sorted(PUBLICATION_ARTIFACTS))
def test_no_publication_artifact_survives_anywhere_in_the_corpus(settings, artifact_name):
    """Sweep every record, not just the ones previously known to be affected."""

    pattern = PUBLICATION_ARTIFACTS[artifact_name]
    records, _ = extract_recommendations(settings.path("source.pdf_path"), settings)
    offenders = [record.recommendation_id for record in records if pattern.search(record.text)]
    assert offenders == [], f"{artifact_name} survived cleaning in: {offenders}"


def test_no_recommendation_ends_with_a_following_subsection_heading(settings):
    """1.6.6 previously ended '[2015] Testicular cancer', bleeding the next heading in."""

    known_titles = {
        title for ranges in SUBSECTION_RANGES.values() for _, title in ranges
    }
    records, _ = extract_recommendations(settings.path("source.pdf_path"), settings)

    offenders = []
    for record in records:
        stripped = record.text.rstrip().rstrip(".")
        for title in known_titles:
            if stripped.endswith(title):
                offenders.append((record.recommendation_id, title))
    assert offenders == [], f"subsection heading bled into: {offenders}"


def test_cleaning_preserves_clinical_content(settings):
    """Artifact removal must be surgical: thresholds and list structure stay intact."""

    records, _ = extract_recommendations(settings.path("source.pdf_path"), settings)
    by_id = {record.recommendation_id: record for record in records}

    # 1.3.1's bulleted criteria were previously split in two by footer text landing
    # between the first and second bullet.
    colorectal = by_id["1.3.1"].text
    for fragment in (
        "with an abdominal mass, or",
        "with a change in bowel habit, or",
        "with iron-deficiency anaemia, or",
        "aged 40 and over with unexplained weight loss and abdominal pain, or",
    ):
        assert fragment in colorectal, f"lost clinical content: {fragment!r}"

    assert "aged 40 and over and have jaundice" in by_id["1.2.4"].text
    assert "at least 10 micrograms of haemoglobin per gram of faeces" in by_id["1.3.2"].text
    assert "aged 45 and over" in by_id["1.6.6"].text
    assert by_id["1.2.4"].text.endswith("[2015]")
