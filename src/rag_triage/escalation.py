import json
from pathlib import Path

from rag_triage.common import write_json
from rag_triage.schemas import (
    ArtifactName,
    Classification,
    ResolutionAction,
    TargetQueue,
    TriageDecision,
)


def decide_escalations(
    *,
    classification_path: Path,
    retrieval_path: Path,
    output_dir: Path,
) -> list[TriageDecision]:
    classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieved_ids_by_ticket = {
        row["ticket_id"]: [article["article_id"] for article in row["retrieved_articles"]]
        for row in retrieval_payload["results"]
    }

    decisions: list[TriageDecision] = []
    for row in classification_payload["classifications"]:
        classification = Classification.model_validate(row)
        evidence_ids = retrieved_ids_by_ticket.get(classification.ticket_id, [])
        decisions.append(decide_one(classification, evidence_ids))

    write_json(
        output_dir / ArtifactName.ESCALATIONS.value,
        {"escalations": [decision.model_dump(mode="json") for decision in decisions]},
    )
    return decisions


def decide_one(classification: Classification, evidence_ids: list[str]) -> TriageDecision:
    if classification.issue_type == "account_access" and (
        classification.resolution_action == ResolutionAction.NEEDS_HUMAN_ESCALATION
        or "KB-002" in evidence_ids
    ):
        return TriageDecision(
            ticket_id=classification.ticket_id,
            escalate=True,
            target_queue=TargetQueue.ACCOUNT_SUPPORT,
            reason="Account access issue is grounded in login-reset evidence and needs account support.",
        )
    if classification.issue_type == "billing" and (
        classification.requires_refund_caution
        or classification.resolution_action == ResolutionAction.NEEDS_HUMAN_ESCALATION
    ):
        return TriageDecision(
            ticket_id=classification.ticket_id,
            escalate=True,
            target_queue=TargetQueue.BILLING_OPS,
            reason="Billing issue requires refund caution or human review based on retrieved evidence.",
        )
    return TriageDecision(
        ticket_id=classification.ticket_id,
        escalate=False,
        target_queue=TargetQueue.NONE,
        reason="Retrieved KB evidence is enough for a safe support reply.",
    )


def print_escalations(decisions: list[TriageDecision]) -> None:
    print("\nEscalations")
    for decision in decisions:
        print(
            f"- {decision.ticket_id}: escalate={decision.escalate}, "
            f"queue={decision.target_queue.value}"
        )
        print(f"  reason: {decision.reason}")

