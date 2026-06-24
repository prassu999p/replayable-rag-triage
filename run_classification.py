from pathlib import Path

from openai import OpenAI

from rag_triage.classification import (
    classify_tickets,
    ensure_retrieval_exists,
    print_classifications,
)
from rag_triage.config import load_settings


def main() -> None:
    settings = load_settings()
    output_dir = Path(".")
    retrieval_path = ensure_retrieval_exists(output_dir)
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
    print_classifications(classifications)
    print("\nWrote ticket_classification.json and llm_calls.jsonl")


if __name__ == "__main__":
    main()

