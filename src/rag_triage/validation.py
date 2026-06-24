import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_triage.common import write_json
from rag_triage.schemas import (
    ArtifactName,
    Classification,
    DraftResponse,
    Ticket,
    TriageDecision,
)


UNSAFE_PHRASES = [
    "send your password",
    "share your password",
    "provide your password",
    "tell us your password",
    "same-day refund is guaranteed",
    "we guarantee a refund",
]

REQUIRED_ARTIFACTS = [
    ArtifactName.RETRIEVAL_RESULTS.value,
    ArtifactName.TICKET_CLASSIFICATION.value,
    ArtifactName.DRAFT_RESPONSES.value,
    ArtifactName.ESCALATIONS.value,
    ArtifactName.LLM_CALLS.value,
]


def validate_outputs(
    *,
    tickets_path: Path = Path("tickets.json"),
    output_dir: Path = Path("."),
) -> dict[str, Any]:
    errors: list[str] = []
    grounding_flags: list[dict[str, str]] = []

    tickets_payload = load_json(tickets_path, errors)
    ticket_ids = {
        Ticket.model_validate(row).ticket_id
        for row in tickets_payload.get("tickets", [])
    } if tickets_payload else set()

    for artifact in REQUIRED_ARTIFACTS:
        if not (output_dir / artifact).exists():
            errors.append(f"Missing required artifact: {artifact}")

    kb_payload = load_json(Path("kb_articles.json"), errors)
    article_text_by_id = {
        row["article_id"]: f"{row['title']} {row['content']}"
        for row in kb_payload.get("articles", [])
    } if kb_payload else {}
    retrieval = load_json(output_dir / ArtifactName.RETRIEVAL_RESULTS.value, errors)
    classification_payload = load_json(output_dir / ArtifactName.TICKET_CLASSIFICATION.value, errors)
    draft_payload = load_json(output_dir / ArtifactName.DRAFT_RESPONSES.value, errors)
    escalation_payload = load_json(output_dir / ArtifactName.ESCALATIONS.value, errors)

    retrieved_ids_by_ticket = {
        row["ticket_id"]: {article["article_id"] for article in row["retrieved_articles"]}
        for row in retrieval.get("results", [])
    } if retrieval else {}

    validate_ticket_coverage("retrieval_results.json", retrieved_ids_by_ticket.keys(), ticket_ids, errors)

    classifications = validate_rows(
        "ticket_classification.json",
        classification_payload.get("classifications", []) if classification_payload else [],
        Classification,
        errors,
    )
    validate_ticket_coverage(
        "ticket_classification.json",
        [row.ticket_id for row in classifications],
        ticket_ids,
        errors,
    )

    drafts = validate_rows(
        "draft_responses.json",
        draft_payload.get("drafts", []) if draft_payload else [],
        DraftResponse,
        errors,
    )
    validate_ticket_coverage(
        "draft_responses.json",
        [row.ticket_id for row in drafts],
        ticket_ids,
        errors,
    )
    for draft in drafts:
        allowed_citations = retrieved_ids_by_ticket.get(draft.ticket_id, set())
        for citation in draft.citations:
            if citation not in allowed_citations:
                errors.append(f"{draft.ticket_id} cites non-retrieved article {citation}")
        flag_count_before = len(grounding_flags)
        flag_unsafe_text(draft, grounding_flags)
        flag_unsupported_claims(draft, article_text_by_id, grounding_flags)
        for flag in grounding_flags[flag_count_before:]:
            errors.append(f"{draft.ticket_id} failed grounding check: {flag['reason']}")

    escalations = validate_rows(
        "escalations.json",
        escalation_payload.get("escalations", []) if escalation_payload else [],
        TriageDecision,
        errors,
    )
    validate_ticket_coverage(
        "escalations.json",
        [row.ticket_id for row in escalations],
        ticket_ids,
        errors,
    )

    report = {
        "ok": not errors,
        "errors": errors,
        "checked_tickets": sorted(ticket_ids),
        "artifact_count": len(REQUIRED_ARTIFACTS),
        "grounding_flag_count": len(grounding_flags),
    }
    write_json(output_dir / ArtifactName.GROUNDING_FLAGS.value, {"flags": grounding_flags})
    write_json(output_dir / ArtifactName.VALIDATION_REPORT.value, report)
    return report


def load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {path.name}: {exc}")
        return {}


def validate_rows(
    artifact: str,
    rows: list[dict[str, Any]],
    model: type,
    errors: list[str],
) -> list[Any]:
    valid_rows: list[Any] = []
    for row in rows:
        try:
            valid_rows.append(model.model_validate(row))
        except ValidationError as exc:
            errors.append(f"Invalid row in {artifact}: {exc}")
    return valid_rows


def validate_ticket_coverage(
    artifact: str,
    seen_ticket_ids: Any,
    expected_ticket_ids: set[str],
    errors: list[str],
) -> None:
    seen = set(seen_ticket_ids)
    missing = expected_ticket_ids - seen
    extra = seen - expected_ticket_ids
    if missing:
        errors.append(f"{artifact} missing tickets: {sorted(missing)}")
    if extra:
        errors.append(f"{artifact} has unexpected tickets: {sorted(extra)}")


def flag_unsafe_text(draft: DraftResponse, grounding_flags: list[dict[str, str]]) -> None:
    lower_text = draft.draft_response.lower()
    for phrase in UNSAFE_PHRASES:
        if phrase in lower_text:
            grounding_flags.append(
                {
                    "ticket_id": draft.ticket_id,
                    "reason": f"Unsafe or unsupported phrase: {phrase}",
                }
            )


def flag_unsupported_claims(
    draft: DraftResponse,
    article_text_by_id: dict[str, str],
    grounding_flags: list[dict[str, str]],
) -> None:
    cited_text = " ".join(article_text_by_id.get(citation, "") for citation in draft.citations)
    supported_terms = set(content_terms(cited_text))
    unsupported_terms = [
        term
        for term in content_terms(draft.draft_response)
        if term not in supported_terms and term not in SAFE_RESPONSE_TERMS
    ]
    if len(set(unsupported_terms)) >= 3:
        grounding_flags.append(
            {
                "ticket_id": draft.ticket_id,
                "reason": (
                    "Draft contains terms not supported by cited KB text: "
                    + ", ".join(sorted(set(unsupported_terms))[:8])
                ),
            }
        )


SAFE_RESPONSE_TERMS = {
    "again",
    "and",
    "are",
    "because",
    "but",
    "can",
    "cannot",
    "contacting",
    "customer",
    "for",
    "hello",
    "hi",
    "may",
    "not",
    "our",
    "please",
    "reaching",
    "should",
    "still",
    "support",
    "thanks",
    "thank",
    "that",
    "the",
    "this",
    "try",
    "unless",
    "will",
    "with",
    "you",
    "your",
}


def content_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) <= 2 or token in SAFE_RESPONSE_TERMS:
            continue
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        terms.append(token)
    return terms


def print_validation(report: dict[str, Any]) -> None:
    print("\nValidation")
    print(f"- ok={report['ok']}")
    print(f"- checked_tickets={', '.join(report['checked_tickets'])}")
    print(f"- grounding_flags={report['grounding_flag_count']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  error: {error}")
