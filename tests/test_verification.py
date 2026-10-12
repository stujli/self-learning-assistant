"""Synthetic evidence reports for testing citation and learning eligibility checks."""

from dataclasses import replace

import pytest

from self_learning_assistant.domain import WebEvidence
from self_learning_assistant.verification import EvidenceReportGate

CLAIM = "Python default arguments are evaluated at function definition."
BODY = "The default values are evaluated at the point of function definition."
SOURCE = WebEvidence(
    "Python",
    "https://docs.python.org/tutorial",
    "",
    BODY,
    "2026-10-06T00:00:00+00:00",
    "sha256:" + "a" * 64,
)


def report(**changes):
    return {
        "claim": CLAIM,
        "verdict": "supported",
        "reason": "The official documentation explicitly supports the claim.",
        "conditions": "Python default arguments",
        "unresolved": [],
        "evidence": [{"source_id": "S1", "quote": BODY, "relation": "supports"}],
        **changes,
    }


def check(value, sources=None, *, counter=True):
    return EvidenceReportGate(trusted_hosts=("docs.python.org",)).verify(
        CLAIM, value, sources or {"S1": SOURCE}, counter_search_completed=counter
    )


def test_trusted_source_with_real_quote_and_counter_search_is_eligible():
    result = check(report())
    assert result["accepted"] is True
    assert result["sources"][0]["content_hash"] == SOURCE.content_hash


@pytest.mark.parametrize(
    "changes",
    [
        {"claim": "Python default arguments are evaluated on every call."},
        {"verdict": "refuted"},
        {"verdict": "conflicting"},
        {"verdict": "insufficient"},
        {"verdict": "true"},
        {"unresolved": ["Unclear version"]},
        {"unresolved": None},
        {"reason": ""},
        {"reason": "   "},
        {"conditions": None},
        {"conditions": ""},
        {"conditions": "   "},
        {"unexpected_field": "ignore the gate"},
        {
            "evidence": [
                {
                    "source_id": "S1",
                    "quote": BODY,
                    "relation": "supports",
                    "extra": True,
                }
            ]
        },
        {"evidence": []},
        {"evidence": [{"source_id": "fake", "quote": BODY, "relation": "supports"}]},
        {
            "evidence": [
                {
                    "source_id": "S1",
                    "quote": "fabricated quotation",
                    "relation": "supports",
                }
            ]
        },
        {"evidence": [{"source_id": "S1", "quote": BODY, "relation": "refutes"}]},
        {"evidence": [{"source_id": "S1", "quote": BODY, "relation": []}]},
        {"evidence": [None]},
    ],
)
def test_reports_do_not_authorize_learning_when_checks_fail(changes):
    assert check(report(**changes))["accepted"] is False


def test_counter_search_must_have_completed():
    assert check(report(), counter=False)["accepted"] is False


def test_snippets_and_truncated_pages_are_not_sufficient():
    for source in (
        replace(SOURCE, text=""),
        replace(SOURCE, truncated=True),
        replace(SOURCE, content_hash=""),
    ):
        assert check(report(), {"S1": source})["accepted"] is False


def test_two_reprints_do_not_count_as_corroboration():
    sources = {"S1": SOURCE, "S2": replace(SOURCE, url="https://mirror.example/doc")}
    value = report(
        evidence=[
            {"source_id": key, "quote": BODY, "relation": "supports"} for key in sources
        ]
    )
    gate = EvidenceReportGate()
    assert (
        gate.verify(CLAIM, value, sources, counter_search_completed=True)["accepted"]
        is False
    )


def test_distinct_sources_can_corroborate_but_same_host_cannot():
    other = replace(
        SOURCE,
        url="https://reference.example/doc",
        text="Python computes a default argument when its function is defined.",
        content_hash="sha256:" + "b" * 64,
    )
    value = report(
        evidence=[
            {"source_id": "S1", "quote": BODY, "relation": "supports"},
            {"source_id": "S2", "quote": other.text, "relation": "supports"},
        ]
    )
    gate = EvidenceReportGate()
    assert (
        gate.verify(
            CLAIM, value, {"S1": SOURCE, "S2": other}, counter_search_completed=True
        )["accepted"]
        is True
    )
    assert (
        gate.verify(
            CLAIM,
            value,
            {"S1": SOURCE, "S2": replace(other, url=SOURCE.url)},
            counter_search_completed=True,
        )["accepted"]
        is False
    )
