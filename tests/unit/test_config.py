from app.core.config import Settings

BASE = {"database_url": "postgresql+psycopg://u:p@h:5432/d"}


def test_allowed_content_types_parsed_from_comma_string():
    settings = Settings(**BASE, allowed_content_types="application/pdf, IMAGE/PNG ,")
    # Whitespace trimmed, case normalised, empty entries dropped.
    assert settings.allowed_content_type_set == frozenset({"application/pdf", "image/png"})


def test_empty_endpoint_means_real_aws():
    assert Settings(**BASE, aws_endpoint_url="").uses_local_aws is False


def test_endpoint_set_means_localstack():
    assert Settings(**BASE, aws_endpoint_url="http://localstack:4566").uses_local_aws is True


def test_analysis_on_demand_is_on_by_default():
    """The default has to match what every deployment runs.

    A routing flag that defaults false while production sets it true means any
    environment brought up without the variable behaves unlike all the others — and this
    one fails invisibly, because a document that files and stops looks exactly like a
    document that files and is still working. The same mistake `prescriptions_enabled`
    made, pinned here so a future edit has to be deliberate.
    """
    assert Settings(**BASE).analysis_on_demand is True


def test_ideal_ranges_are_on_by_default():
    """The R&D-approved ideal range is the product; the printed range is the fallback.

    Off by default meant the code that reads the whole THP catalogue had never run in
    production, and the evidence was `ai_thp_fallbacks` sitting empty — a table that
    exists to tell R&D which parameters still need curating, answering a question nothing
    had ever asked it. Pinned here so a future edit has to be deliberate.
    """
    assert Settings(**BASE).ideal_ranges_enabled is True
