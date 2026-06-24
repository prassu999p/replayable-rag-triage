from pathlib import Path

from rag_triage.validation import print_validation, validate_outputs


if __name__ == "__main__":
    print_validation(validate_outputs(tickets_path=Path("tickets.json"), output_dir=Path(".")))

