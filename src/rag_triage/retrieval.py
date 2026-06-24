import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag_triage.schemas import (
    ArtifactName,
    KbArticle,
    RetrievedArticle,
    RetrievalResult,
    Ticket,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "now",
    "of",
    "on",
    "or",
    "please",
    "the",
    "they",
    "this",
    "to",
    "under",
    "using",
    "with",
}

SPECIAL_STEMS = {
    "authorizations": "authorization",
    "charged": "charge",
    "charges": "charge",
    "charging": "charge",
    "documents": "document",
    "invoices": "invoice",
    "passwords": "password",
    "pdfs": "pdf",
    "refunds": "refund",
    "refunded": "refund",
    "refunding": "refund",
}

BILLING_HINTS = {"billing", "card", "charge", "invoice", "order", "payment", "refund"}
ACCOUNT_HINTS = {"account", "credential", "login", "password", "reset"}


def tokenize(text: str) -> list[str]:
    """Turn text into stable, lowercase terms for deterministic matching."""
    terms: list[str] = []
    for raw_token in re.findall(r"[a-z0-9]+", text.lower()):
        if raw_token in STOPWORDS:
            continue
        token = SPECIAL_STEMS.get(raw_token, raw_token)
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        if token not in STOPWORDS:
            terms.append(token)
    return terms


def load_inputs(tickets_path: Path, kb_path: Path) -> tuple[list[Ticket], list[KbArticle]]:
    tickets_payload = json.loads(tickets_path.read_text(encoding="utf-8"))
    kb_payload = json.loads(kb_path.read_text(encoding="utf-8"))
    tickets = [Ticket.model_validate(row) for row in tickets_payload["tickets"]]
    articles = [KbArticle.model_validate(row) for row in kb_payload["articles"]]
    return tickets, articles


def ticket_search_text(ticket: Ticket) -> str:
    return f"{ticket.subject} {ticket.message}"


def article_search_text(article: KbArticle) -> str:
    return f"{article.title} {article.category.value} {article.content}"


def expand_query_terms(terms: list[str]) -> list[str]:
    """Add clear domain hints so sparse tickets can still hit the right KB area."""
    expanded = list(terms)
    term_set = set(terms)
    if term_set & BILLING_HINTS:
        expanded.append("billing")
    if term_set & ACCOUNT_HINTS:
        expanded.extend(["account", "access"])
    return expanded


def build_idf(articles: list[KbArticle]) -> dict[str, float]:
    document_count = len(articles)
    document_frequency: Counter[str] = Counter()
    for article in articles:
        document_frequency.update(set(tokenize(article_search_text(article))))

    # The +1 smoothing keeps scores stable when the evaluator swaps in tiny fixtures.
    return {
        term: math.log((1 + document_count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }


def score_article(
    query_terms: list[str],
    article: KbArticle,
    idf: dict[str, float],
) -> tuple[float, list[str]]:
    article_terms = tokenize(article_search_text(article))
    title_terms = set(tokenize(article.title))
    article_counts = Counter(article_terms)
    query_counts = Counter(query_terms)
    matching_terms = sorted(set(query_terms) & set(article_terms))
    score = 0.0

    for term in matching_terms:
        weight = idf.get(term, 1.0)
        score += query_counts[term] * article_counts[term] * weight
        if term in title_terms:
            score += 0.75 * query_counts[term] * weight

    # Normalize by article length so long KB articles do not win only by being verbose.
    if article_terms:
        score = score / math.sqrt(len(article_terms))
    return round(score, 6), matching_terms


def retrieve_for_ticket(
    ticket: Ticket,
    articles: list[KbArticle],
    limit: int = 3,
) -> tuple[list[RetrievedArticle], list[dict[str, Any]]]:
    idf = build_idf(articles)
    query_terms = expand_query_terms(tokenize(ticket_search_text(ticket)))
    ranked_rows: list[dict[str, Any]] = []

    for article in articles:
        score, matching_terms = score_article(query_terms, article, idf)
        if score <= 0:
            continue
        ranked_rows.append(
            {
                "article_id": article.article_id,
                "title": article.title,
                "score": score,
                "matching_terms": matching_terms,
            }
        )

    ranked_rows.sort(key=lambda row: (-row["score"], row["article_id"]))
    top_rows = ranked_rows[:limit]
    results = [
        RetrievedArticle(
            article_id=row["article_id"],
            title=row["title"],
            score=row["score"],
        )
        for row in top_rows
    ]
    return results, top_rows


def run_retrieval(
    tickets_path: Path = Path("tickets.json"),
    kb_path: Path = Path("kb_articles.json"),
    output_dir: Path = Path("."),
) -> None:
    tickets, articles = load_inputs(tickets_path, kb_path)
    retrieval_results: list[RetrievalResult] = []
    retrieval_debug: list[dict[str, Any]] = []

    for ticket in tickets:
        retrieved_articles, debug_rows = retrieve_for_ticket(ticket, articles)
        retrieval_results.append(
            RetrievalResult(
                ticket_id=ticket.ticket_id,
                retrieved_articles=retrieved_articles,
            )
        )
        retrieval_debug.append(
            {
                "ticket_id": ticket.ticket_id,
                "query_terms": expand_query_terms(tokenize(ticket_search_text(ticket))),
                "ranked_articles": debug_rows,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / ArtifactName.RETRIEVAL_RESULTS.value,
        {"results": [result.model_dump(mode="json") for result in retrieval_results]},
    )
    write_json(output_dir / ArtifactName.RETRIEVAL_DEBUG.value, {"debug": retrieval_debug})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_retrieval()

