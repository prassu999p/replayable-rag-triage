import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_triage.retrieval import load_inputs, run_retrieval
from rag_triage.schemas import (
    ArtifactName,
    Classification,
    KbArticle,
    LlmCallLog,
    Ticket,
)
from rag_triage.common import write_json


CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ticket_id": {"type": "string"},
        "issue_type": {
            "type": "string",
            "enum": ["billing", "account_access", "technical_problem", "how_to", "other"],
        },
        "urgency": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
        "resolution_action": {
            "type": "string",
            "enum": ["answer_from_kb", "needs_human_escalation", "request_more_info"],
        },
        "customer_sensitive": {"type": "boolean"},
        "requires_refund_caution": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "ticket_id",
        "issue_type",
        "urgency",
        "resolution_action",
        "customer_sensitive",
        "requires_refund_caution",
        "rationale",
    ],
}


def ensure_retrieval_exists(output_dir: Path) -> Path:
    retrieval_path = output_dir / ArtifactName.RETRIEVAL_RESULTS.value
    if not retrieval_path.exists():
        run_retrieval(output_dir=output_dir)
    return retrieval_path


def make_classification_prompt(ticket: Ticket, articles: list[KbArticle]) -> str:
    article_blocks = "\n\n".join(
        [
            (
                f"Article {article.article_id}\n"
                f"Title: {article.title}\n"
                f"Category: {article.category.value}\n"
                f"Content: {article.content}"
            )
            for article in articles
        ]
    )
    return f"""Classify this support ticket using only the ticket and retrieved KB evidence.

Do not invent categories. Use exactly these allowed values:
- issue_type: billing | account_access | technical_problem | how_to | other
- urgency: critical | high | medium | low
- resolution_action: answer_from_kb | needs_human_escalation | request_more_info

Set customer_sensitive to true for billing, account access, refunds, credentials, payment, privacy, or account-security issues.
Set requires_refund_caution to true when the ticket asks for, implies, or pressures for a refund or duplicate-charge decision.

Ticket:
- ticket_id: {ticket.ticket_id}
- customer_tier: {ticket.customer_tier}
- language: {ticket.language}
- subject: {ticket.subject}
- message: {ticket.message}
- created_at: {ticket.created_at}

Retrieved KB articles:
{article_blocks}
"""


def classify_tickets(
    *,
    client: Any,
    model: str,
    tickets_path: Path,
    kb_path: Path,
    retrieval_path: Path,
    output_dir: Path,
    provider: str,
) -> list[Classification]:
    tickets, articles = load_inputs(tickets_path, kb_path)
    article_by_id = {article.article_id: article for article in articles}
    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieved_ids_by_ticket = {
        row["ticket_id"]: [article["article_id"] for article in row["retrieved_articles"][:3]]
        for row in retrieval_payload["results"]
    }

    classifications: list[Classification] = []
    log_path = output_dir / ArtifactName.LLM_CALLS.value
    log_path.write_text("", encoding="utf-8")

    for ticket in tickets:
        top_articles = [
            article_by_id[article_id]
            for article_id in retrieved_ids_by_ticket.get(ticket.ticket_id, [])
            if article_id in article_by_id
        ]
        prompt = make_classification_prompt(ticket, top_articles)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful support triage classifier. Return only the "
                        "structured JSON requested by the schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            # Structured output makes the model return JSON that matches the schema.
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ticket_classification",
                    "schema": CLASSIFICATION_SCHEMA,
                    "strict": True,
                }
            },
        )
        classification = Classification.model_validate_json(extract_output_text(response))
        classifications.append(classification)
        append_llm_log(
            log_path=log_path,
            ticket_id=ticket.ticket_id,
            provider=provider,
            model=model,
            prompt=prompt,
        )

    write_json(
        output_dir / ArtifactName.TICKET_CLASSIFICATION.value,
        {"classifications": [row.model_dump(mode="json") for row in classifications]},
    )
    return classifications


def extract_output_text(response: Any) -> str:
    if hasattr(response, "output_text"):
        return response.output_text
    if isinstance(response, dict) and "output_text" in response:
        return str(response["output_text"])
    raise TypeError("OpenAI response did not contain output_text")


def append_llm_log(
    *,
    log_path: Path,
    ticket_id: str,
    provider: str,
    model: str,
    prompt: str,
) -> None:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    record = LlmCallLog(
        stage="classification",
        ticket_id=ticket_id,
        timestamp=datetime.now(UTC).isoformat(),
        provider=provider,
        model=model,
        prompt_hash=prompt_hash,
        input_artifacts=[
            "tickets.json",
            "kb_articles.json",
            ArtifactName.RETRIEVAL_RESULTS.value,
        ],
        output_artifact=ArtifactName.TICKET_CLASSIFICATION.value,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def print_classifications(classifications: list[Classification]) -> None:
    print("Ticket classifications")
    for row in classifications:
        print(
            f"- {row.ticket_id}: issue={row.issue_type.value}, urgency={row.urgency.value}, "
            f"action={row.resolution_action.value}, sensitive={row.customer_sensitive}, "
            f"refund_caution={row.requires_refund_caution}"
        )
        print(f"  rationale: {row.rationale}")
