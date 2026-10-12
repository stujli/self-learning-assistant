from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from urllib.error import HTTPError

from self_learning_assistant.product.research import Researcher, bounded
from self_learning_assistant.product.retrieval import Retrieval, failure
from self_learning_assistant.tools.domain import (
    FetchedPage,
    SearchHit,
    SearchResponse,
    WebToolError,
)

CLAIM = "Python default argument values are evaluated at function definition."
BODY = "Python default argument values are evaluated at function definition, not again on each call."
PLAN = {
    "neutral": [
        "Python default argument evaluation",
        "Python default values independent source",
    ],
    "counter": [
        "Python default argument counterexamples",
        "Python default values exceptions",
    ],
    "preferred_hosts": [],
}


def hit(host: str, index: int = 0, *, related: bool = True) -> SearchHit:
    return SearchHit(
        "Python default argument evaluation" if related else "Python homepage",
        f"https://{host}/page-{index}",
        CLAIM if related else "Welcome to Python",
        "fixture",
    )


class Web:
    def __init__(
        self,
        neutral: list[SearchHit],
        counter: list[SearchHit],
        *,
        targeted: list[SearchHit] | None = None,
    ) -> None:
        self.neutral, self.counter = neutral, counter
        self.targeted = targeted if targeted is not None else neutral
        self.searches: list[str] = []
        self.fetches: list[tuple[str, int]] = []
        self.failed_hosts: set[str] = set()
        self.body = BODY

    def search(self, query: str, *, max_results: int = 8) -> SearchResponse:
        self.searches.append(query)
        pool = (
            self.targeted
            if query.startswith("site:")
            else self.counter
            if "counterexamples" in query or "exceptions" in query
            else self.neutral
        )
        return SearchResponse(query, "fixture", tuple(pool[:max_results]))

    def fetch(self, url: str, *, max_chars: int = 20_000) -> FetchedPage:
        from urllib.parse import urlsplit

        self.fetches.append((url, max_chars))
        if urlsplit(url).hostname in self.failed_hosts:
            raise WebToolError("blocked") from HTTPError(
                url, 403, "Forbidden", None, None
            )
        text = self.body + "\nSource: " + url
        return FetchedPage(
            url,
            url,
            "Python documentation",
            text[:max_chars],
            "2026-10-10T00:00:00Z",
            "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            "text/plain",
            len(text) > max_chars,
        )


def retrieve(
    web: Web, *, preferred: list[str] | None = None, trusted: set[str] | None = None
) -> Retrieval:
    plan = {**PLAN, "preferred_hosts": preferred or []}
    result = Retrieval(
        web, CLAIM, plan, trusted or set(), time.monotonic() + 5, bounded
    )
    result.run()
    return result


def test_counter_results_cannot_exhaust_neutral_budget() -> None:
    web = Web(
        [hit(f"neutral-{i}.example.org") for i in range(8)],
        [hit(f"counter-{i}.example.org") for i in range(8)],
    )
    result = retrieve(web)
    assert len(web.fetches) == 8
    assert result.lanes["neutral"] == 4
    assert result.lanes["counter"] == 2
    assert result.lanes["supplement"] == 2
    assert all("neutral-" in web.fetches[i][0] for i in (0, 2, 4, 5))
    assert all("counter-" in web.fetches[i][0] for i in (1, 3))
    assert len(web.searches) <= 4


def test_official_site_search_and_relevance_precede_unrelated_homepage() -> None:
    homepage = hit("docs.python.org", related=False)
    official = hit("docs.python.org", 1)
    related = hit("other.example.org")
    web = Web([homepage, related, official], [hit("counter.example.org")])
    result = retrieve(web, trusted={"docs.python.org"})
    assert web.searches[0].startswith("site:docs.python.org ")
    assert web.fetches[0] == (official.url, 50_000)
    ranking = [item.url for item in result.ranked()]
    assert ranking.index(related.url) < ranking.index(homepage.url)


def test_generic_official_matches_do_not_displace_specific_independent_result() -> None:
    generic = SearchHit(
        "Python time values",
        "https://docs.python.org/3/library/time.html",
        "",
        "fixture",
    )
    specific = hit("independent.example.org")
    web = Web([generic, specific], [])
    result = retrieve(web, trusted={"docs.python.org"})
    assert web.fetches[0][0] == specific.url
    generic_diagnostic = next(
        item for item in result.diagnostic()["candidates"] if item["url"] == generic.url
    )
    assert generic_diagnostic["matched_terms"] == ["python", "values"]
    assert not generic_diagnostic["relevant"]
    assert result.minimum_overlap == 4


def test_weak_official_metadata_does_not_suppress_authority_supplement() -> None:
    generic = SearchHit(
        "Python time values",
        "https://docs.python.org/3/library/time.html",
        "",
        "fixture",
    )
    # Even an accidentally related body does not make a weak search hit strong.
    web = Web([generic], [])
    retrieve(web, preferred=["docs.python.org", "second.example.org"])
    assert web.searches[-1].startswith("site:second.example.org ")


def test_duplicate_urls_merge_purposes_and_are_fetched_once() -> None:
    page = hit("docs.python.org")
    alias = SearchHit(page.title, page.url + "#section", CLAIM, "fixture")
    result = retrieve(Web([page], [alias]), preferred=["docs.python.org"])
    assert result.fetch_count == 1
    assert len(result.candidates) == 1
    candidate = next(iter(result.candidates.values()))
    assert candidate.purposes == {"neutral", "counter"}
    assert len(candidate.query_ids) >= 2


def test_unused_lane_budget_transfers_without_exceeding_limits() -> None:
    web = Web([hit(f"neutral-{i}.example.org") for i in range(8)], [])
    result = retrieve(web)
    assert result.counter_search
    assert result.fetch_count == 8
    assert result.lanes["neutral"] == 4
    assert result.lanes["supplement"] == 4


def test_fourth_search_refills_unused_counter_quota() -> None:
    class LateCounter(Web):
        def search(self, query: str, **options: Any) -> SearchResponse:
            if query == PLAN["counter"][1]:
                self.counter = [hit(f"late-counter-{i}.example.org") for i in range(3)]
            return super().search(query, **options)

    web = LateCounter([hit(f"neutral-{i}.example.org") for i in range(8)], [])
    result = retrieve(web)
    assert len(web.searches) == 4
    assert result.fetch_count == 8
    assert result.lanes["neutral"] == 4
    assert result.lanes["counter"] == 2
    assert result.lanes["supplement"] == 2


def test_blocked_host_stops_after_two_failures_and_preserves_diagnostics() -> None:
    web = Web(
        [hit("blocked.example.org", i) for i in range(8)]
        + [hit("healthy.example.org")],
        [],
    )
    web.failed_hosts.add("blocked.example.org")
    result = retrieve(web, trusted={"blocked.example.org"})
    failures = [
        event
        for event in result.events
        if event["phase"] == "fetch" and not event["ok"]
    ]
    assert len(failures) == 2
    assert all(
        event["status_code"] == 403 and event["error_kind"] == "http"
        for event in failures
    )
    assert any(
        item.skip_reason == "host_failures" for item in result.candidates.values()
    )
    assert result.fetch_count <= 8


def test_host_caps_allow_source_diversity() -> None:
    web = Web(
        [hit("docs.python.org", i) for i in range(6)],
        [hit("counter.example.org", i) for i in range(6)],
    )
    result = retrieve(web, preferred=["docs.python.org"])
    assert sum("docs.python.org" in url for url, _ in web.fetches) == 3
    assert sum("counter.example.org" in url for url, _ in web.fetches) == 2
    assert any(item.skip_reason == "host_budget" for item in result.candidates.values())


def test_fourth_search_targets_missing_second_authority() -> None:
    web = Web([hit("general.example.org")], [hit("counter.example.org")], targeted=[])
    result = retrieve(web, preferred=["first.example.org", "second.example.org"])
    assert len(web.searches) == 4
    assert web.searches[-1].startswith("site:second.example.org ")
    assert result.preferred == ["first.example.org", "second.example.org"]


def test_invalid_suggestions_do_not_become_search_targets() -> None:
    web = Web([hit("general.example.org")], [])
    result = retrieve(web, preferred=["http://127.0.0.1/private", "127.0.0.1"])
    assert not result.preferred
    assert all(not query.startswith("site:") for query in web.searches)
    assert result.diagnostic()["trusted_hosts"] == []


def test_source_excerpt_budget_retains_each_source_and_exact_text() -> None:
    web = Web(
        [hit(f"neutral-{i}.example.org") for i in range(8)],
        [hit(f"counter-{i}.example.org") for i in range(8)],
    )
    web.body = "\n".join(
        f"Python default argument values are evaluated at function definition in example {i}."
        for i in range(40)
    )
    result = retrieve(web)
    passages = result.passages()
    assert len(passages) == 32
    assert {source for source, _ in passages.values()} == set(result.sources)
    assert all(
        12 <= len(text) <= 1000 and text in result.sources[source].text
        for source, text in passages.values()
    )


def test_excerpt_preserves_wrapped_conditions_and_version_numbers() -> None:
    web = Web([hit("docs.python.org")], [])
    web.body = (
        "Python 3.14 default argument values are evaluated once,\n"
        "when the function is defined, not on each call.\n"
        "A proposed alternative syntax evaluates defaults at call time."
    )
    result = retrieve(web)
    spans = [text for _, text in result.passages().values()]
    assert any(
        "Python 3.14" in text and "not on each call" in text and "\n" in text
        for text in spans
    )
    assert all(text in web.body + "\nSource: " + web.neutral[0].url for text in spans)


class Planner:
    def generate_json(
        self, messages: Any, schema: Any, **options: Any
    ) -> dict[str, Any]:
        if "RESEARCH_QUERY_PLAN" in messages[-1]["content"]:
            return {**PLAN, "preferred_hosts": ["docs.python.org"]}
        payload = json.loads(
            messages[-1]["content"].rsplit("```json\n", 1)[1].rsplit("\n```", 1)[0]
        )
        passage = payload["passages"][0]
        return {
            "claim": payload["claim"],
            "verdict": "supported",
            "reason": "Direct statement",
            "conditions": "Python",
            "unresolved": [],
            "evidence": [
                {
                    "source_id": passage["source_id"],
                    "passage_id": passage["passage_id"],
                    "relation": "supports",
                }
            ],
        }

    def generate(self, messages: Any, **options: Any) -> str:
        return ""


def test_model_preferred_host_does_not_grant_single_source_trust() -> None:
    page = hit("docs.python.org")
    report = Researcher(Planner(), Web([page], [page])).verify(CLAIM)
    assert report["accepted"] is False
    assert report["retrieval"]["preferred_hosts"] == ["docs.python.org"]
    assert report["retrieval"]["trusted_hosts"] == []
    trusted = Researcher(
        Planner(), Web([page], [page]), trusted_hosts=["docs.python.org"]
    ).verify(CLAIM)
    assert trusted["accepted"] is True


def test_authority_long_page_is_complete_before_citation_validation() -> None:
    web = Web([hit("docs.python.org")], [])
    web.body = (
        "\n".join(f"Unrelated introductory passage number {i}." for i in range(600))
        + "\n"
        + BODY
    )
    report = Researcher(Planner(), web, trusted_hosts=["docs.python.org"]).verify(CLAIM)
    assert len(web.body) > 20_000
    assert report["accepted"], report
    assert not report["captured_sources"]["S1"]["truncated"]
    assert BODY in report["captured_sources"]["S1"]["text"]


def test_unknown_failure_is_not_fabricated_http_status() -> None:
    diagnostic = failure(WebToolError("message containing HTTP 403"))
    assert diagnostic["error_kind"] == "unknown"
    assert "status_code" not in diagnostic


def test_retrieval_deadline_leaves_time_for_report() -> None:
    class Slow(Web):
        def search(self, query: str, **options: Any) -> SearchResponse:
            time.sleep(0.2)
            return super().search(query)

    web = Slow([], [])
    result = Retrieval(web, CLAIM, PLAN, set(), time.monotonic() + 0.02, bounded)
    started = time.monotonic()
    result.run()
    assert time.monotonic() - started < 0.15
    assert result.fetch_count == 0
    assert result.events[0]["error_kind"] == "deadline"
