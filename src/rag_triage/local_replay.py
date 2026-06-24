import hashlib
from pathlib import Path

from rag_triage.common import write_json
from rag_triage.drafting import make_drafting_prompt
from rag_triage.retrieval import load_inputs
from rag_triage.schemas import (
    ArtifactName,
    Classification,
    DraftResponse,
    IssueType,
    LlmCallLog,
    ResolutionAction,
    Urgency,
)


REPLAY_TIMESTAMP = "1970-01-01T00:00:00+00:00"
REPLAY_MODEL = "deterministic-rules-v1"
REPLAY_PROVIDER = "local_replay"


def run_local_replay_ai(
    *,
    tickets_path: Path,
    kb_path: Path,
    retrieval_path: Path,
    output_dir: Path,
) -> tuple[list[Classification], list[DraftResponse]]:
    tickets, articles = load_inputs(tickets_path, kb_path)
    article_by_id = {article.article_id: article for article in articles}
    retrieval_payload = __import__("json").loads(retrieval_path.read_text(encoding="utf-8"))
    retrieved_ids_by_ticket = {
        row["ticket_id"]: [article["article_id"] for article in row["retrieved_articles"]]
        for row in retrieval_payload["results"]
    }

    classifications: list[Classification] = []
    drafts: list[DraftResponse] = []
    log_path = output_dir / ArtifactName.LLM_CALLS.value
    log_path.write_text("", encoding="utf-8")

    for ticket in tickets:
        top_articles = [article_by_id[article_id] for article_id in retrieved_ids_by_ticket[ticket.ticket_id]]
        classification = classify_locally(ticket_id=ticket.ticket_id, text=f"{ticket.subject} {ticket.message}")
        classifications.append(classification)
        prompt = f"local classification\n{ticket.model_dump_json()}\n{[a.article_id for a in top_articles]}"
        append_replay_log(log_path, "classification", ticket.ticket_id, prompt, ArtifactName.TICKET_CLASSIFICATION.value)

        draft = draft_locally(classification, [article.article_id for article in top_articles])
        drafts.append(draft)
        draft_prompt = make_drafting_prompt(ticket, classification, top_articles)
        append_replay_log(log_path, "drafting", ticket.ticket_id, draft_prompt, ArtifactName.DRAFT_RESPONSES.value)

    write_json(
        output_dir / ArtifactName.TICKET_CLASSIFICATION.value,
        {"classifications": [row.model_dump(mode="json") for row in classifications]},
    )
    write_json(
        output_dir / ArtifactName.DRAFT_RESPONSES.value,
        {"drafts": [row.model_dump(mode="json") for row in drafts]},
    )
    return classifications, drafts


def classify_locally(*, ticket_id: str, text: str) -> Classification:
    lower_text = text.lower()
    if "password" in lower_text or "login" in lower_text:
        return Classification(
            ticket_id=ticket_id,
            issue_type=IssueType.ACCOUNT_ACCESS,
            urgency=Urgency.MEDIUM,
            resolution_action=ResolutionAction.NEEDS_HUMAN_ESCALATION,
            customer_sensitive=True,
            requires_refund_caution=False,
            rationale="Replay mode detected a login or password-reset access issue.",
        )
    if "invoice" in lower_text or "download" in lower_text:
        return Classification(
            ticket_id=ticket_id,
            issue_type=IssueType.HOW_TO,
            urgency=Urgency.LOW,
            resolution_action=ResolutionAction.ANSWER_FROM_KB,
            customer_sensitive=False,
            requires_refund_caution=False,
            rationale="Replay mode detected a how-to question answered by the KB.",
        )
    if "charge" in lower_text or "refund" in lower_text:
        return Classification(
            ticket_id=ticket_id,
            issue_type=IssueType.BILLING,
            urgency=Urgency.MEDIUM,
            resolution_action=ResolutionAction.ANSWER_FROM_KB,
            customer_sensitive=True,
            requires_refund_caution=True,
            rationale="Replay mode detected duplicate-charge or refund language.",
        )
    return Classification(
        ticket_id=ticket_id,
        issue_type=IssueType.OTHER,
        urgency=Urgency.LOW,
        resolution_action=ResolutionAction.REQUEST_MORE_INFO,
        customer_sensitive=False,
        requires_refund_caution=False,
        rationale="Replay mode could not match the ticket to a specific support area.",
    )


def draft_locally(classification: Classification, retrieved_ids: list[str]) -> DraftResponse:
    if classification.issue_type == IssueType.ACCOUNT_ACCESS:
        return DraftResponse(
            ticket_id=classification.ticket_id,
            draft_response=(
                "Thanks for reaching out. Please verify you are using the most recent "
                "password, clear cached sessions, and try again in an incognito window. "
                "Because the reset succeeded but login still fails, we will escalate this "
                "to account support."
            ),
            citations=[article_id for article_id in ["KB-002"] if article_id in retrieved_ids],
        )
    if classification.issue_type == IssueType.HOW_TO:
        return DraftResponse(
            ticket_id=classification.ticket_id,
            draft_response=(
                "You can download invoice PDFs from Billing > Documents. Filter by month "
                "and download each invoice as a PDF."
            ),
            citations=[article_id for article_id in ["KB-003"] if article_id in retrieved_ids],
        )
    if classification.issue_type == IssueType.BILLING:
        return DraftResponse(
            ticket_id=classification.ticket_id,
            draft_response=(
                "Thanks for contacting us. You may see both a completed charge and a "
                "temporary pending authorization. Pending authorizations usually disappear "
                "within 3-5 business days. We cannot promise a refund unless two settled "
                "charges are confirmed."
            ),
            citations=[article_id for article_id in ["KB-001", "KB-004"] if article_id in retrieved_ids],
        )
    return DraftResponse(
        ticket_id=classification.ticket_id,
        draft_response="Thanks for contacting us. We need a little more information to route this safely.",
        citations=retrieved_ids[:1],
    )


def append_replay_log(
    log_path: Path,
    stage: str,
    ticket_id: str,
    prompt: str,
    output_artifact: str,
) -> None:
    record = LlmCallLog(
        stage=stage,
        ticket_id=ticket_id,
        timestamp=REPLAY_TIMESTAMP,
        provider=REPLAY_PROVIDER,
        model=REPLAY_MODEL,
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        input_artifacts=["tickets.json", "kb_articles.json", ArtifactName.RETRIEVAL_RESULTS.value],
        output_artifact=output_artifact,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")

