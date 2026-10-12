# Knowledge base

The knowledge base stores copies of text, Markdown, and PDF documents, and retrieves relevant passages to assist answers. Imported documents and verified knowledge saved by [web verification](web-verification.md) have different sources and statuses. Importing a document does not verify its facts.

## Importing and using documents

```bash
uv run sla knowledge import /absolute/path/to/notes.md
uv run sla knowledge list
uv run sla knowledge search "connection pool"
uv run sla knowledge show DOCUMENT_ID
uv run sla chat "What is the connection pool size in the document?" --no-learn --offline
```

Replace DOCUMENT_ID with an ID returned by list or search. Chat automatically retrieves relevant passages. Citations include the document ID and original line numbers or page numbers; show lets you inspect the document and its locations.

| Input | Limits |
| --- | --- |
| UTF-8 text, Markdown | At most 20 MB per file and 2 million extracted characters |
| PDF | The same file and extraction limits apply; at most 500 pages, with extractable text |
| Scanned or encrypted PDF | OCR and encrypted PDF extraction are not supported |

Content is deduplicated. `knowledge delete DOCUMENT_ID` removes the application copy and search index while keeping the original file. Deleting a knowledge source used in training also revokes related versions; see [Background learning](background-learning.md#version-management).

## Retrieval and citations

Imported documents are stored locally, and relevant passages are retrieved when answering. Chinese and English queries are supported without downloading an additional retrieval model. Response language still follows the current question or configured preference.

Private imports do not automatically become verified knowledge or parameter training sources. Accepted stable public knowledge must also meet learning eligibility requirements before training; see [Background learning](background-learning.md#training-and-evaluation).

Search results are relevant candidates, and the model can still omit citations or misread the text. Complete citations alone do not guarantee a correct answer. Check important conclusions against the original document or perform public web verification.
