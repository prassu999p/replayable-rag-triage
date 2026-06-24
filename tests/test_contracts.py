from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_triage.config import Settings, load_settings
from rag_triage.schemas import (
    ArtifactName,
    Classification,
    IssueType,
    KbArticle,
    PipelineStage,
    ResolutionAction,
    TargetQueue,
    Ticket,
    TriageDecision,
    Urgency,
)


def test_load_settings_reads_api_key_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-key\nAI_PROVIDER=openai\nAI_MODEL=gpt-4.1-mini\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.openai_api_key == "test-key"
    assert settings.ai_provider == "openai"
    assert settings.ai_model == "gpt-4.1-mini"


def test_settings_rejects_missing_api_key() -> None:
    with pytest.raises(ValidationError):
        Settings(openai_api_key="")


def test_ticket_and_kb_article_shapes_match_input_files() -> None:
    ticket = Ticket(
        ticket_id="T-1001",
        customer_tier="standard",
        language="en",
        subject="I was charged twice for one order",
        message="One order, two visible charges.",
        created_at="2026-05-20T10:15:00Z",
    )
    article = KbArticle(
        article_id="KB-001",
        title="Duplicate card charges and pending authorizations",
        category=IssueType.BILLING,
        updated_at="2026-05-01T09:00:00Z",
        content="Pending authorizations usually disappear within 3-5 business days.",
    )

    assert ticket.ticket_id == "T-1001"
    assert article.category == IssueType.BILLING


def test_classification_shape_enforces_controlled_taxonomy() -> None:
    classification = Classification(
        ticket_id="T-1001",
        issue_type=IssueType.BILLING,
        urgency=Urgency.MEDIUM,
        resolution_action=ResolutionAction.ANSWER_FROM_KB,
        customer_sensitive=True,
        requires_refund_caution=True,
        rationale="Duplicate charge language requires refund caution.",
    )

    assert classification.issue_type == IssueType.BILLING

    with pytest.raises(ValidationError):
        Classification(
            ticket_id="T-1001",
            issue_type="payments",
            urgency="medium",
            resolution_action="answer_from_kb",
            customer_sensitive=True,
            requires_refund_caution=True,
            rationale="Invalid category should fail.",
        )


def test_triage_decision_requires_none_queue_when_not_escalated() -> None:
    decision = TriageDecision(
        ticket_id="T-1003",
        escalate=False,
        target_queue=TargetQueue.NONE,
        reason="The KB fully answers the customer question.",
    )

    assert decision.target_queue == TargetQueue.NONE

    with pytest.raises(ValidationError):
        TriageDecision(
            ticket_id="T-1003",
            escalate=False,
            target_queue=TargetQueue.BILLING_OPS,
            reason="Non-escalated tickets must use none.",
        )


def test_pipeline_stages_and_artifact_names_cover_requirement() -> None:
    assert [stage.value for stage in PipelineStage] == [
        "INIT",
        "INPUTS_LOADED",
        "KB_INDEXED",
        "TICKETS_NORMALISED",
        "EVIDENCE_RETRIEVED",
        "TICKETS_CLASSIFIED",
        "RESPONSES_DRAFTED",
        "ESCALATIONS_DECIDED",
        "OUTPUTS_VALIDATED",
        "RESULTS_FINALISED",
    ]
    assert ArtifactName.RETRIEVAL_RESULTS.value == "retrieval_results.json"
    assert ArtifactName.LLM_CALLS.value == "llm_calls.jsonl"

