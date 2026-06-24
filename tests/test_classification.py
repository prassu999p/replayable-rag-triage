import json
from pathlib import Path

from rag_triage.classification import classify_tickets, make_classification_prompt
from rag_triage.retrieval import load_inputs, run_retrieval


class FakeResponse:
    output_text = json.dumps(
        {
            "ticket_id": "T-1001",
            "issue_type": "billing",
            "urgency": "medium",
            "resolution_action": "answer_from_kb",
            "customer_sensitive": True,
            "requires_refund_caution": True,
            "rationale": "The ticket concerns a duplicate card charge and refund caution.",
        }
    )


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_make_classification_prompt_includes_ticket_docs_and_taxonomy() -> None:
    tickets, articles = load_inputs(Path("tickets.json"), Path("kb_articles.json"))
    prompt = make_classification_prompt(tickets[0], articles[:3])

    assert "T-1001" in prompt
    assert "KB-001" in prompt
    assert "billing | account_access | technical_problem | how_to | other" in prompt
    assert "Do not invent categories" in prompt


def test_classify_tickets_writes_validated_artifact_and_llm_log(tmp_path: Path) -> None:
    run_retrieval(output_dir=tmp_path)
    client = FakeOpenAIClient()

    classifications = classify_tickets(
        client=client,
        model="test-model",
        tickets_path=Path("tickets.json"),
        kb_path=Path("kb_articles.json"),
        retrieval_path=tmp_path / "retrieval_results.json",
        output_dir=tmp_path,
        provider="openai",
    )

    output = json.loads((tmp_path / "ticket_classification.json").read_text())
    log_lines = (tmp_path / "llm_calls.jsonl").read_text().strip().splitlines()

    assert len(classifications) == 3
    assert output["classifications"][0]["issue_type"] == "billing"
    assert output["classifications"][0]["requires_refund_caution"] is True
    assert len(log_lines) == 3
    assert json.loads(log_lines[0])["stage"] == "classification"
    assert client.responses.calls[0]["text"]["format"]["strict"] is True

