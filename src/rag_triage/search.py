import argparse
from pathlib import Path

from rag_triage.retrieval import load_inputs, retrieve_for_ticket


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the top three KB docs for a ticket.")
    parser.add_argument("ticket_id", help="Ticket ID to search, for example T-1001.")
    parser.add_argument("--tickets", default="tickets.json", help="Path to tickets.json.")
    parser.add_argument("--kb", default="kb_articles.json", help="Path to kb_articles.json.")
    args = parser.parse_args()

    tickets, articles = load_inputs(Path(args.tickets), Path(args.kb))
    ticket_by_id = {ticket.ticket_id: ticket for ticket in tickets}
    if args.ticket_id not in ticket_by_id:
        raise SystemExit(f"Ticket not found: {args.ticket_id}")

    results, debug_rows = retrieve_for_ticket(ticket_by_id[args.ticket_id], articles, limit=3)

    print(f"Top 3 documents for {args.ticket_id}")
    if not results:
        print("No useful KB match found.")
        return

    for rank, (article, debug_row) in enumerate(zip(results, debug_rows, strict=True), start=1):
        terms = ", ".join(debug_row["matching_terms"]) or "none"
        print(f"{rank}. {article.article_id} | score={article.score:.6f} | {article.title}")
        print(f"   matching_terms: {terms}")


if __name__ == "__main__":
    main()

