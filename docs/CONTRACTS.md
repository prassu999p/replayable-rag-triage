# Project Contracts

## Input Files

The pipeline must read these files from disk:

- `tickets.json`
- `kb_articles.json`

The evaluator may replace both files with equivalent fixtures using the same structure.

## Ticket Shape

```json
{
  "ticket_id": "T-1001",
  "customer_tier": "standard",
  "language": "en",
  "subject": "I was charged twice for one order",
  "message": "I placed one order yesterday but my card shows two charges.",
  "created_at": "2026-05-20T10:15:00Z"
}
```

## Knowledge Base Article Shape

```json
{
  "article_id": "KB-001",
  "title": "Duplicate card charges and pending authorizations",
  "category": "billing",
  "updated_at": "2026-05-01T09:00:00Z",
  "content": "Customers may see both a completed charge and a temporary pending authorization."
}
```

## Controlled Taxonomy

Allowed issue types:

- `billing`
- `account_access`
- `technical_problem`
- `how_to`
- `other`

Allowed urgency levels:

- `critical`
- `high`
- `medium`
- `low`

Allowed resolution actions:

- `answer_from_kb`
- `needs_human_escalation`
- `request_more_info`

Allowed target queues:

- `billing_ops`
- `account_support`
- `general_support`
- `none`

## Classification Shape

```json
{
  "ticket_id": "T-1001",
  "issue_type": "billing",
  "urgency": "medium",
  "resolution_action": "answer_from_kb",
  "customer_sensitive": true,
  "requires_refund_caution": true,
  "rationale": "Duplicate charge language requires refund caution."
}
```

## Draft Response Shape

```json
{
  "ticket_id": "T-1001",
  "draft_response": "Thanks for contacting us. A pending authorization may disappear within 3-5 business days.",
  "citations": ["KB-001"]
}
```

## Triage Decision Shape

```json
{
  "ticket_id": "T-1001",
  "escalate": true,
  "target_queue": "billing_ops",
  "reason": "The ticket asks for a refund and the KB requires caution for duplicate charges."
}
```

When `escalate` is `false`, `target_queue` must be `none`.

## Required Stage Order

```text
INIT
 -> INPUTS_LOADED
 -> KB_INDEXED
 -> TICKETS_NORMALISED
 -> EVIDENCE_RETRIEVED
 -> TICKETS_CLASSIFIED
 -> RESPONSES_DRAFTED
 -> ESCALATIONS_DECIDED
 -> OUTPUTS_VALIDATED
 -> RESULTS_FINALISED
```

Response drafting must not run before retrieval and classification. Escalation decisions must be grounded in classification output and retrieved evidence.

## Artifact Names

Required:

- `retrieval_results.json`
- `ticket_classification.json`
- `draft_responses.json`
- `escalations.json`
- `validation_report.json`
- `llm_calls.jsonl`

Attempted extras:

- `retrieval_debug.json`
- `grounding_flags.json`

## Retrieval Behavior

The current retriever is deterministic and local. It lowercases text, removes common stopwords, applies a tiny set of support-domain stems, expands obvious billing and account-access hints, and scores each KB article with stable TF-IDF-style weights.

Ties are broken by `article_id`, so the same inputs produce the same ranked outputs. Retrieval output includes an `input_hash` derived from `tickets.json` and `kb_articles.json`; the main pipeline reuses retrieval artifacts when this hash still matches and regenerates them when inputs change.

Generate retrieval artifacts with:

```bash
.venv/bin/python run_retrieval.py
```

Inspect one ticket with:

```bash
.venv/bin/python -m rag_triage.search T-1001
```

## Classification Behavior

The current classifier sends each ticket and its top three retrieved KB articles to OpenAI. It requests strict structured JSON and validates the returned object against the controlled taxonomy in `src/rag_triage/schemas.py`.

Run classification with:

```bash
.venv/bin/python run_classification.py
```

This writes:

- `ticket_classification.json`
- `llm_calls.jsonl`

Each `llm_calls.jsonl` row records the stage, ticket ID, timestamp, provider, model, prompt hash, input artifacts, and output artifact.

## Full Pipeline

Run the live OpenAI-backed pipeline with:

```bash
.venv/bin/python run_pipeline.py
```

Run the deterministic replay pipeline with:

```bash
.venv/bin/python run_pipeline.py --replay
```

Replay mode intentionally avoids network calls. It uses local deterministic classification and drafting rules, fixed timestamps, sorted JSON keys, and the same input files to produce identical artifacts across reruns.

The full pipeline uses `PipelineStateMachine` to enforce the required stage order at runtime. Attempting to advance to a stage out of order raises an error instead of silently continuing.

## Validation Behavior

Validation checks:

- required artifacts exist
- JSON files parse
- every ticket appears in each artifact
- classification values match the controlled taxonomy
- draft citations only cite retrieved article IDs
- draft wording is checked against cited KB text so unsupported concrete claims are flagged
- escalations obey the `target_queue` rule
- unsafe phrases such as asking for passwords or guaranteeing refunds are flagged
