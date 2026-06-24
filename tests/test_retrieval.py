import json
import subprocess
from pathlib import Path

from rag_triage.retrieval import (
    load_inputs,
    retrieve_for_ticket,
    run_retrieval,
    tokenize,
)


def test_tokenize_normalizes_words_for_replayable_matching() -> None:
    assert tokenize("Charges, charged, and charging twice!") == [
        "charge",
        "charge",
        "charge",
        "twice",
    ]


def test_retrieve_for_ticket_returns_top_three_in_stable_order() -> None:
    tickets, articles = load_inputs(Path("tickets.json"), Path("kb_articles.json"))

    results, debug_rows = retrieve_for_ticket(tickets[0], articles, limit=3)

    assert [article.article_id for article in results] == ["KB-001", "KB-004", "KB-003"]
    assert results[0].score > results[1].score
    assert debug_rows[0]["article_id"] == "KB-001"
    assert "charge" in debug_rows[0]["matching_terms"]


def test_run_retrieval_writes_required_artifacts(tmp_path: Path) -> None:
    run_retrieval(
        tickets_path=Path("tickets.json"),
        kb_path=Path("kb_articles.json"),
        output_dir=tmp_path,
    )

    retrieval_results = json.loads((tmp_path / "retrieval_results.json").read_text())
    retrieval_debug = json.loads((tmp_path / "retrieval_debug.json").read_text())

    assert [row["ticket_id"] for row in retrieval_results["results"]] == [
        "T-1001",
        "T-1002",
        "T-1003",
    ]
    assert retrieval_results["results"][1]["retrieved_articles"][0]["article_id"] == "KB-002"
    assert retrieval_debug["debug"][2]["ranked_articles"][0]["article_id"] == "KB-003"


def test_search_command_prints_top_three_docs() -> None:
    completed = subprocess.run(
        [".venv/bin/python", "-m", "rag_triage.search", "T-1001"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Top 3 documents for T-1001" in completed.stdout
    assert "1. KB-001" in completed.stdout
    assert "2. KB-004" in completed.stdout
    assert "3. KB-003" in completed.stdout

