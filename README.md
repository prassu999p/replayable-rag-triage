# Replayable RAG Triage

This project is a small replayable RAG pipeline for support-ticket triage. It loads tickets and KB articles, retrieves grounded evidence, classifies each ticket, drafts customer replies, decides escalations, validates outputs, and writes inspectable artifacts.

The full pipeline enforces the required stage order at runtime:

```text
INIT -> INPUTS_LOADED -> KB_INDEXED -> TICKETS_NORMALISED -> EVIDENCE_RETRIEVED -> TICKETS_CLASSIFIED -> RESPONSES_DRAFTED -> ESCALATIONS_DECIDED -> OUTPUTS_VALIDATED -> RESULTS_FINALISED
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

For live OpenAI mode, create a local `.env` file from `.env.example` and replace the placeholder API key:

```bash
cp .env.example .env
```

`.env` is ignored by git. The project loads `OPENAI_API_KEY`, `AI_PROVIDER`, and `AI_MODEL` from that file via `rag_triage.config.load_settings`.

Replay mode does not need an API key.

## Verify

```bash
.venv/bin/python -m pytest -q
```

Expected result:

```text
19 passed
```

## Run Retrieval

Generate deterministic retrieval artifacts only:

```bash
.venv/bin/python run_retrieval.py
```

This writes:

- `retrieval_results.json`
- `retrieval_debug.json`

Search one ticket and print its top three KB documents:

```bash
.venv/bin/python -m rag_triage.search T-1001
```

Expected top match for the sample data:

```text
1. KB-001 | ... | Duplicate card charges and pending authorizations
```

## Run Full Pipeline

Put a real API key in `.env`:

```text
OPENAI_API_KEY=your-real-key
AI_PROVIDER=openai
AI_MODEL=gpt-4.1-mini
```

Then run the live OpenAI-backed pipeline:

```bash
.venv/bin/python run_pipeline.py
```

This writes:

- `retrieval_results.json`
- `retrieval_debug.json`
- `ticket_classification.json`
- `draft_responses.json`
- `escalations.json`
- `grounding_flags.json`
- `validation_report.json`
- `llm_calls.jsonl`

Expected terminal summary:

```text
Ticket classifications
Draft responses
Escalations
Validation
- ok=True
Wrote all pipeline artifacts.
```

## Replay Mode

Replay mode does not call OpenAI. It uses deterministic local classification and drafting rules plus fixed timestamps in `llm_calls.jsonl`, so rerunning with the same inputs produces identical artifacts.

```bash
.venv/bin/python run_pipeline.py --replay
```

Use this command when you want a clean, reproducible demo without network access or API cost.

The main pipeline also skips retrieval recomputation when `tickets.json` and `kb_articles.json` have not changed. Retrieval artifacts store an input hash and are regenerated when either input file changes.

Expected validation summary:

```text
Validation
- ok=True
- checked_tickets=T-1001, T-1002, T-1003
- grounding_flags=0
```

## Validate Existing Outputs

```bash
.venv/bin/python validate.py
```

## Current Files

- `src/rag_triage/config.py`: loads API configuration from `.env`.
- `src/rag_triage/schemas.py`: defines tickets, KB articles, retrieval results, classification, draft responses, triage decisions, LLM call logs, artifacts, stages, and controlled taxonomies.
- `src/rag_triage/retrieval.py`: deterministic tokenization, scoring, ranking, and retrieval artifact writing.
- `src/rag_triage/stage_machine.py`: runtime enforcement for the required pipeline stage order.
- `src/rag_triage/classification.py`: builds grounded classification prompts, calls OpenAI with strict structured output, validates taxonomy, logs calls, and prints results.
- `src/rag_triage/drafting.py`: builds grounded drafting prompts, calls OpenAI with strict structured output, logs calls, and writes draft replies.
- `src/rag_triage/escalation.py`: decides target queues from classification and retrieved evidence.
- `src/rag_triage/validation.py`: validates required artifacts, ticket coverage, taxonomy, citations, escalation rules, unsafe draft phrases, and whether draft claims are supported by cited KB text.
- `src/rag_triage/local_replay.py`: deterministic replay classifier and drafter for identical reruns without network calls.
- `src/rag_triage/search.py`: command-line helper for inspecting the top three KB matches for a ticket.
- `tests/test_contracts.py`: verifies the project contracts.
- `tests/test_retrieval.py`: verifies deterministic retrieval, artifact writing, and CLI output.
- `tickets.json`: sample support tickets.
- `kb_articles.json`: sample knowledge base articles.
- `docs/CONTRACTS.md`: compact reference for shapes, taxonomy, stage order, and artifact names.
