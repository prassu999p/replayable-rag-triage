# Implementation Phases

The main implementation risk is building AI calls before the replayable pipeline spine exists. The project is structured around three phases.

## Phase 1: Deterministic Retrieval

Build the non-LLM RAG foundation first.

Do this next:

- Load `tickets.json` and `kb_articles.json`
- Normalize ticket text and KB text
- Build deterministic keyword or TF-IDF retrieval
- Retrieve top 3 articles per ticket
- Save:
  - `retrieval_results.json`
  - `retrieval_debug.json`

Success criteria: rerunning retrieval with the same inputs gives byte-stable or logically identical results.

Implemented replay polish also stores an input hash in `retrieval_results.json`, allowing the main pipeline to skip retrieval recomputation when `tickets.json` and `kb_articles.json` are unchanged.

## Phase 2: Structured AI Stages

Add classification and response drafting after retrieval works.

Do this after retrieval:

- Create an LLM wrapper that reads API config from `.env`
- Log every call to `llm_calls.jsonl`
- Classify each ticket using retrieved KB evidence
- Validate taxonomy strictly
- Draft grounded customer replies using only retrieved articles and classification
- Save:
  - `ticket_classification.json`
  - `draft_responses.json`

Success criteria: every ticket has valid structured output, citations only reference retrieved KB IDs, and prompts are traceable.

## Phase 3: Escalation, Validation, Replay Polish

Finish the production-minded parts.

Do this last:

- Decide escalation from classification + retrieved evidence
- Add grounding and unsafe-content checks
- Implement `validate.py`
- Generate:
  - `escalations.json`
  - `grounding_flags.json`
  - `validation_report.json`
- Update `README.md` with exact run commands

Success criteria: evaluator can delete generated artifacts, run the pipeline from clean inputs, then run validation and inspect every intermediate file.

Runtime polish includes explicit stage-order enforcement with `PipelineStateMachine` and a KB-content grounding check that flags concrete draft terms unsupported by cited articles.
