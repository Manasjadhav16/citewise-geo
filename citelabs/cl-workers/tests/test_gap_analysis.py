import pytest

from app.gap_analysis import EvidenceSource, Gap, find_content_gaps


def test_stub_raises_until_implemented():
    source = EvidenceSource(
        url="https://rival.com/fees",
        domain="rival.com",
        category="direct_competitor",
        competitor_group="direct_business",
        evidence="- UPI fee: 0% for merchants under 2,000 INR",
        stability=0.67,
    )
    with pytest.raises(NotImplementedError):
        find_content_gaps("- Razorpay charges 2% per transaction", [source], query="UPI fees")


def test_gap_shape():
    gap = Gap(query="UPI fees", topic="Zero-MDR UPI", claims=["UPI fee: 0%"], supporting_sources=["https://rival.com/fees"])
    assert gap.weight == 0.0 and gap.sandbox_partial_match is None
