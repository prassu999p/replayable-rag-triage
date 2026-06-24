import json
import subprocess
from pathlib import Path

from rag_triage.escalation import decide_escalations
from rag_triage.retrieval import run_retrieval
from rag_triage.validation import validate_outputs


def write_phase3_inputs(output_dir: Path) -> None:
    run_retrieval(output_dir=output_dir)
    (output_dir / "ticket_classification.json").write_text(
        json.dumps(
            {
                "classifications": [
                    {
                        "ticket_id": "T-1001",
                        "issue_type": "billing",
                        "urgency": "medium",
                        "resolution_action": "answer_from_kb",
                        "customer_sensitive": True,
                        "requires_refund_caution": True,
                        "rationale": "Duplicate card charge needs refund caution.",
                    },
                    {
                        "ticket_id": "T-1002",
                        "issue_type": "account_access",
                        "urgency": "medium",
                        "resolution_action": "needs_human_escalation",
                        "customer_sensitive": True,
                        "requires_refund_caution": False,
                        "rationale": "Login still fails after password reset.",
                    },
                    {
                        "ticket_id": "T-1003",
                        "issue_type": "how_to",
                        "urgency": "low",
                        "resolution_action": "answer_from_kb",
                        "customer_sensitive": False,
                        "requires_refund_caution": False,
                        "rationale": "The KB explains invoice PDF downloads.",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "draft_responses.json").write_text(
        json.dumps(
            {
                "drafts": [
                    {
                        "ticket_id": "T-1001",
                        "draft_response": "A pending authorization may disappear in 3-5 business days. We cannot promise a refund unless two settled charges are confirmed.",
                        "citations": ["KB-001", "KB-004"],
                    },
                    {
                        "ticket_id": "T-1002",
                        "draft_response": "Please retry with your most recent password, clear cached sessions, and try an incognito window. We will escalate this to account support.",
                        "citations": ["KB-002"],
                    },
                    {
                        "ticket_id": "T-1003",
                        "draft_response": "Invoice PDFs are available under Billing > Documents. You can filter by month and download each invoice.",
                        "citations": ["KB-003"],
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_decide_escalations_uses_classification_and_evidence(tmp_path: Path) -> None:
    write_phase3_inputs(tmp_path)

    decisions = decide_escalations(
        classification_path=tmp_path / "ticket_classification.json",
        retrieval_path=tmp_path / "retrieval_results.json",
        output_dir=tmp_path,
    )

    assert decisions[0].target_queue == "billing_ops"
    assert decisions[1].target_queue == "account_support"
    assert decisions[2].target_queue == "none"


def test_validate_outputs_accepts_complete_safe_pipeline(tmp_path: Path) -> None:
    write_phase3_inputs(tmp_path)
    decide_escalations(
        classification_path=tmp_path / "ticket_classification.json",
        retrieval_path=tmp_path / "retrieval_results.json",
        output_dir=tmp_path,
    )
    (tmp_path / "llm_calls.jsonl").write_text("", encoding="utf-8")

    report = validate_outputs(
        tickets_path=Path("tickets.json"),
        output_dir=tmp_path,
    )

    assert report["ok"] is True
    assert report["errors"] == []
    assert (tmp_path / "validation_report.json").exists()
    assert (tmp_path / "grounding_flags.json").exists()


def test_replay_command_produces_identical_artifacts() -> None:
    command = [".venv/bin/python", "run_pipeline.py", "--replay"]

    subprocess.run(command, check=True, capture_output=True, text=True)
    first = Path("ticket_classification.json").read_text(encoding="utf-8")
    first_drafts = Path("draft_responses.json").read_text(encoding="utf-8")
    first_logs = Path("llm_calls.jsonl").read_text(encoding="utf-8")

    subprocess.run(command, check=True, capture_output=True, text=True)

    assert Path("ticket_classification.json").read_text(encoding="utf-8") == first
    assert Path("draft_responses.json").read_text(encoding="utf-8") == first_drafts
    assert Path("llm_calls.jsonl").read_text(encoding="utf-8") == first_logs

