import argparse
from pathlib import Path

from openai import OpenAI

from rag_triage.classification import classify_tickets, print_classifications
from rag_triage.config import load_settings
from rag_triage.drafting import draft_responses, print_drafts
from rag_triage.escalation import decide_escalations, print_escalations
from rag_triage.local_replay import (
    run_local_replay_classifications,
    run_local_replay_drafts,
)
from rag_triage.retrieval import ensure_retrieval_current
from rag_triage.schemas import ArtifactName, PipelineStage
from rag_triage.stage_machine import PipelineStateMachine
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
    machine = PipelineStateMachine()
    machine.advance(PipelineStage.INPUTS_LOADED)
    retrieval_path = ensure_retrieval_current(output_dir=output_dir)
    machine.advance(PipelineStage.KB_INDEXED)
    machine.advance(PipelineStage.TICKETS_NORMALISED)
    machine.advance(PipelineStage.EVIDENCE_RETRIEVED)

    if args.replay:
        classifications = run_local_replay_classifications(
            tickets_path=Path("tickets.json"),
            kb_path=Path("kb_articles.json"),
            retrieval_path=retrieval_path,
            output_dir=output_dir,
        )
        machine.advance(PipelineStage.TICKETS_CLASSIFIED)
        drafts = run_local_replay_drafts(
            tickets_path=Path("tickets.json"),
            kb_path=Path("kb_articles.json"),
            retrieval_path=retrieval_path,
            output_dir=output_dir,
            classifications=classifications,
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
        machine.advance(PipelineStage.TICKETS_CLASSIFIED)
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
    machine.advance(PipelineStage.RESPONSES_DRAFTED)

    decisions = decide_escalations(
        classification_path=output_dir / ArtifactName.TICKET_CLASSIFICATION.value,
        retrieval_path=retrieval_path,
        output_dir=output_dir,
    )
    machine.advance(PipelineStage.ESCALATIONS_DECIDED)
    report = validate_outputs(tickets_path=Path("tickets.json"), output_dir=output_dir)
    machine.advance(PipelineStage.OUTPUTS_VALIDATED)
    machine.advance(PipelineStage.RESULTS_FINALISED)

    print_classifications(classifications)
    print_drafts(drafts)
    print_escalations(decisions)
    print_validation(report)
    print("\nWrote all pipeline artifacts.")


if __name__ == "__main__":
    main()
