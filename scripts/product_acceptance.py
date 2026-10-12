"""Real-model release gates. Fixture and live runs are explicitly distinguished."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from self_learning_assistant.product.config import Paths, Settings
from self_learning_assistant.product.model import create_model
from self_learning_assistant.product.research import Researcher, web_service
from self_learning_assistant.tools.domain import FetchedPage, SearchHit, SearchResponse


class FixtureWeb:
    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def search(self, query: str, *, max_results: int = 4) -> SearchResponse:
        return SearchResponse(
            query,
            "fixture",
            tuple(
                SearchHit(
                    "Fixture", f"https://fixtures.example.org/{index}", "", "fixture"
                )
                for index, _ in enumerate(self.case["bodies"])
            ),
        )

    def fetch(self, url: str, *, max_chars: int = 20_000) -> FetchedPage:
        import hashlib

        text = self.case["bodies"][int(url.rsplit("/", 1)[1])]
        return FetchedPage(
            url,
            url,
            "Authored evaluation fixture",
            text,
            "2026-10-09T00:00:00+00:00",
            "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            "text/plain",
            False,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a user-supplied evaluation dataset",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--app-home", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--reserve", type=float, default=60)
    args = parser.parse_args()
    if not args.app_home.is_absolute():
        parser.error("app-home must be absolute")
    if not args.dataset.is_file():
        parser.error("Dataset is unavailable; provide an existing --dataset file")
    settings = Settings(
        Paths(*(args.app_home / part for part in ("config", "data", "state", "cache")))
    )
    if not args.fixture and not settings.effective()["online"]:
        parser.error("Live acceptance requires explicitly enabled online configuration")
    model = create_model(settings, name=args.model)
    cases = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ][: args.limit]
    if not cases:
        parser.error("Dataset contains no cases")
    if not args.fixture and any(
        not case.get("ground_truth_sources") or not case.get("reviewed_at")
        for case in cases
    ):
        parser.error("Live cases require ground_truth_sources and reviewed_at")
    results: list[dict[str, Any]] = []
    for case in cases:
        web: Any = FixtureWeb(case) if args.fixture else web_service(settings)
        trusted = (
            ["fixtures.example.org"] if args.fixture else case.get("trusted_hosts", [])
        )
        report = Researcher(
            model,
            web,
            trusted_hosts=trusted,
            timeout=args.timeout,
            reserve=args.reserve,
        ).verify(case["claim"])
        result = {"case_id": case["id"], "expected": case["expected"], "report": report}
        results.append(result)
        summary = summarize(results, args.fixture, len(cases))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "expected": case["expected"],
                    "verdict": report["verdict"],
                    "accepted": report["accepted"],
                    "seconds": report["execution"]["seconds"],
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary["gate_passed"] else 3


def summarize(
    results: list[dict[str, Any]], fixture: bool, planned: int
) -> dict[str, Any]:
    positives = [item for item in results if item["expected"] == "supported"]
    negatives = [item for item in results if item["expected"] != "supported"]
    counts = {
        kind: sum(item["expected"] == kind for item in results)
        for kind in ("supported", "refuted", "insufficient", "conflicting")
    }
    false_accepts = sum(bool(item["report"]["accepted"]) for item in negatives)
    supported_rate = sum(bool(item["report"]["accepted"]) for item in positives) / max(
        1, len(positives)
    )
    failed = sum(bool(item["report"]["execution"]["failed"]) for item in results)
    return {
        "version": 1,
        "mode": "fixture-real-model" if fixture else "live-real-model",
        "created_at": time.time(),
        "planned": planned,
        "completed": len(results),
        "counts": counts,
        "false_accepts": false_accepts,
        "supported_accept_rate": supported_rate,
        "execution_failures": failed,
        "verdict_accuracy": sum(
            item["report"]["verdict"] == item["expected"] for item in results
        )
        / max(1, len(results)),
        "gate_passed": len(results) == 40
        and all(count == 10 for count in counts.values())
        and false_accepts == 0
        and supported_rate >= 0.8
        and failed == 0,
        "results": results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
