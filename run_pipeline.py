import argparse
from pathlib import Path

from openai import OpenAI

from rag_triage.classification import classify_tickets, print_classifications
from rag_triage.config import load_settings
from rag_triage.drafting import draft_responses, print_drafts
from rag_triage.escalation import decide_escalations, print_escalations
from rag_triage.local_replay import run_local_replay_ai
from rag_triage.retrieval import run_retrieval
from rag_triage.schemas import ArtifactName
from rag_triage.validation import print_validation, validate_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the replayable RAG triage pipeline.")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use deterministic local classification and drafting instead of OpenAI.",
    )
    args = parser.parse_args()

    output_dir = Path(".")
    run_retrieval(output_dir=output_dir)
    retrieval_path = output_dir / ArtifactName.RETRIEVAL_RESULTS.value

    if args.replay:
        classifications, drafts = run_local_replay_ai(
            tickets_path=Path("tickets.json"),
            kb_path=Path("kb_articles.json"),
            retrieval_path=retrieval_path,
            output_dir=output_dir,
        )
    else:
        settings = load_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        classifications = classify_tickets(
            client=client,
            model=settings.ai_model,
            tickets_path=Path("tickets.json"),
            kb_path=Path("kb_articles.json"),
            retrieval_path=retrieval_path,
            output_dir=output_dir,
            provider=settings.ai_provider,
        )
        drafts = draft_responses(
            client=client,
            model=settings.ai_model,
            tickets_path=Path("tickets.json"),
            kb_path=Path("kb_articles.json"),
            retrieval_path=retrieval_path,
            classification_path=output_dir / ArtifactName.TICKET_CLASSIFICATION.value,
            output_dir=output_dir,
            provider=settings.ai_provider,
        )

    decisions = decide_escalations(
        classification_path=output_dir / ArtifactName.TICKET_CLASSIFICATION.value,
        retrieval_path=retrieval_path,
        output_dir=output_dir,
    )
    report = validate_outputs(tickets_path=Path("tickets.json"), output_dir=output_dir)

    print_classifications(classifications)
    print_drafts(drafts)
    print_escalations(decisions)
    print_validation(report)
    print("\nWrote all pipeline artifacts.")


if __name__ == "__main__":
    main()

