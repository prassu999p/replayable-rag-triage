import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_triage.classification import extract_output_text
from rag_triage.common import write_json
from rag_triage.retrieval import load_inputs
from rag_triage.schemas import (
    ArtifactName,
    Classification,
    DraftResponse,
    KbArticle,
    LlmCallLog,
    Ticket,
)


DRAFT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ticket_id": {"type": "string"},
        "draft_response": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["ticket_id", "draft_response", "citations"],
}


def make_drafting_prompt(
    ticket: Ticket,
    classification: Classification,
    articles: list[KbArticle],
) -> str:
    article_blocks = "\n\n".join(
        [
            (
                f"Article {article.article_id}\n"
                f"Title: {article.title}\n"
                f"Content: {article.content}"
            )
            for article in articles
        ]
    )
    return f"""Draft a customer-facing support reply grounded only in the classification and retrieved KB articles.

Rules:
- Answer in the ticket language: {ticket.language}
- Do not promise refunds or actions not supported by the KB.
- Do not ask for passwords or unsafe secret information.
- If the classification says escalation is needed, mention escalation clearly.
- Use citations only from the retrieved article IDs shown below.

Ticket:
- ticket_id: {ticket.ticket_id}
- subject: {ticket.subject}
- message: {ticket.message}

Classification:
- issue_type: {classification.issue_type.value}
- urgency: {classification.urgency.value}
- resolution_action: {classification.resolution_action.value}
- customer_sensitive: {classification.customer_sensitive}
- requires_refund_caution: {classification.requires_refund_caution}
- rationale: {classification.rationale}

Retrieved KB articles:
{article_blocks}
"""


def draft_responses(
    *,
    client: Any,
    model: str,
    tickets_path: Path,
    kb_path: Path,
    retrieval_path: Path,
    classification_path: Path,
    output_dir: Path,
    provider: str,
) -> list[DraftResponse]:
    tickets, articles = load_inputs(tickets_path, kb_path)
    ticket_by_id = {ticket.ticket_id: ticket for ticket in tickets}
    article_by_id = {article.article_id: article for article in articles}
    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))

    retrieved_ids_by_ticket = {
        row["ticket_id"]: [article["article_id"] for article in row["retrieved_articles"][:3]]
        for row in retrieval_payload["results"]
    }
    classifications = [
        Classification.model_validate(row)
        for row in classification_payload["classifications"]
    ]
    drafts: list[DraftResponse] = []
    log_path = output_dir / ArtifactName.LLM_CALLS.value

    for classification in classifications:
        ticket = ticket_by_id[classification.ticket_id]
        top_articles = [
            article_by_id[article_id]
            for article_id in retrieved_ids_by_ticket.get(ticket.ticket_id, [])
            if article_id in article_by_id
        ]
        prompt = make_drafting_prompt(ticket, classification, top_articles)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You draft concise, safe support replies grounded only in supplied KB text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            # Strict structured output gives us predictable keys to validate downstream.
            text={
                "format": {
                    "type": "json_schema",
                    "name": "draft_response",
                    "schema": DRAFT_RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
        )
        draft = DraftResponse.model_validate_json(extract_output_text(response))
        drafts.append(draft)
        append_drafting_log(
            log_path=log_path,
            ticket_id=ticket.ticket_id,
            provider=provider,
            model=model,
            prompt=prompt,
        )

    write_json(
        output_dir / ArtifactName.DRAFT_RESPONSES.value,
        {"drafts": [row.model_dump(mode="json") for row in drafts]},
    )
    return drafts


def append_drafting_log(
    *,
    log_path: Path,
    ticket_id: str,
    provider: str,
    model: str,
    prompt: str,
) -> None:
    record = LlmCallLog(
        stage="drafting",
        ticket_id=ticket_id,
        timestamp=datetime.now(UTC).isoformat(),
        provider=provider,
        model=model,
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        input_artifacts=[
            "tickets.json",
            "kb_articles.json",
            ArtifactName.RETRIEVAL_RESULTS.value,
            ArtifactName.TICKET_CLASSIFICATION.value,
        ],
        output_artifact=ArtifactName.DRAFT_RESPONSES.value,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def print_drafts(drafts: list[DraftResponse]) -> None:
    print("\nDraft responses")
    for draft in drafts:
        print(f"- {draft.ticket_id}: citations={', '.join(draft.citations)}")
        print(f"  {draft.draft_response}")

