from ng12_rag.ingestion import (
    expected_recommendation_ids,
    extract_recommendations,
    validate_source,
)


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
