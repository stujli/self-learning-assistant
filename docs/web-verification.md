# Web verification

Web verification checks public claims and saves queries, original webpage text, citations, and applicability conditions in an inspectable report. Networking is disabled by default. Chat remains available offline if search is unavailable, and results with insufficient evidence are not saved as verified knowledge.

## Search and networking configuration

`uv run sla setup` can connect to an existing SearXNG instance or request confirmation to start a local container through Docker Compose. To deploy manually:

```bash
docker compose -f infra/searxng/compose.yml up -d
curl 'http://127.0.0.1:8080/search?q=test&format=json'
uv run sla config set search.searxng_url http://127.0.0.1:8080
uv run sla doctor --check-search
uv run sla config set online true
```

The container exposes only `127.0.0.1:8080`, and SearXNG must enable JSON output. The wizard's deployment file is stored at `search/compose.yml` under the configuration directory; use the corresponding file to manage its container. Turning off networking in the application does not stop the container automatically. For long-term deployment, use a pinned image version you have verified.

Optional Brave Search uses `SLA_BRAVE_API_KEY`, which is not stored in configuration or backups. In auto mode it can be used as a fallback when SearXNG fails or returns no results. You can connect to an existing search service without Docker.

## Verification commands

```bash
uv run sla research "Python default arguments are evaluated at function definition time." --web --trusted-host docs.python.org --json
uv run sla chat "Please verify this public claim" --web
uv run sla chat "What are the related public facts?" --web --web-query "A query that may be sent publicly"
```

`--web` permits networking for this operation only, without changing global settings. research can omit web when online=true. `--offline` overrides networking for the current operation. Automatic verification keeps personal questions local; supply an explicit web-query to check related public facts. Private-mode rules are described in [Privacy and data](privacy-and-data.md#session-controls).

research supports `--trusted-host HOST`, `--timeout 300`, `--report-reserve 60`, and `--output FILE`. trusted_hosts must contain exact hostnames explicitly chosen by the user; model-suggested domains do not automatically become trusted sources.

`learn topic TEXT` can create a task containing up to three claims, which the [learning queue](background-learning.md#learning-queue) verifies individually. A topic task marked verified means only that at least one claim passed; inspect reports for the individual results.

## Verification process and limits

Verification searches public material, performs neutral, counter-evidence, and supplementary retrieval, and generates a report from the original text. Each claim permits at most four searches and eight fetches, including failed attempts. Reports contain citations and applicability conditions.

Reports must cite captured original text, complete counter-evidence searches, and address conflicts. A trusted primary source or two supporting sources with different hostnames and different text meet the source acceptance threshold. Semantic judgments still depend on the model and are not absolute proof of truth. Only accepted results are saved to the [knowledge base](knowledge-base.md); insufficient evidence leaves a deferred result.

Fetching accepts only public HTTP(S) targets and rejects private networks, loopback or special-purpose addresses, and URL credentials. JavaScript is not executed. Downloads are limited to 2 MB, truncated text cannot serve as a qualifying citation, and queries suspected of containing credentials are rejected.

In JSON reports, retrieval records retrieval budgets and skip reasons, while events records queries, fetches, and failures. `--output FILE` saves the complete report and source snapshots for checking conclusions. Search service and webpage availability affect verification results.
