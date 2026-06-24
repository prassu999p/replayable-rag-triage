from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class IssueType(StrEnum):
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    TECHNICAL_PROBLEM = "technical_problem"
    HOW_TO = "how_to"
    OTHER = "other"


class Urgency(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResolutionAction(StrEnum):
    ANSWER_FROM_KB = "answer_from_kb"
    NEEDS_HUMAN_ESCALATION = "needs_human_escalation"
    REQUEST_MORE_INFO = "request_more_info"


class TargetQueue(StrEnum):
    BILLING_OPS = "billing_ops"
    ACCOUNT_SUPPORT = "account_support"
    GENERAL_SUPPORT = "general_support"
    NONE = "none"


class PipelineStage(StrEnum):
    INIT = "INIT"
    INPUTS_LOADED = "INPUTS_LOADED"
    KB_INDEXED = "KB_INDEXED"
    TICKETS_NORMALISED = "TICKETS_NORMALISED"
    EVIDENCE_RETRIEVED = "EVIDENCE_RETRIEVED"
    TICKETS_CLASSIFIED = "TICKETS_CLASSIFIED"
    RESPONSES_DRAFTED = "RESPONSES_DRAFTED"
    ESCALATIONS_DECIDED = "ESCALATIONS_DECIDED"
    OUTPUTS_VALIDATED = "OUTPUTS_VALIDATED"
    RESULTS_FINALISED = "RESULTS_FINALISED"


class ArtifactName(StrEnum):
    RETRIEVAL_RESULTS = "retrieval_results.json"
    RETRIEVAL_DEBUG = "retrieval_debug.json"
    TICKET_CLASSIFICATION = "ticket_classification.json"
    DRAFT_RESPONSES = "draft_responses.json"
    ESCALATIONS = "escalations.json"
    VALIDATION_REPORT = "validation_report.json"
    GROUNDING_FLAGS = "grounding_flags.json"
    LLM_CALLS = "llm_calls.jsonl"


class Ticket(BaseModel):
    ticket_id: str = Field(min_length=1)
    customer_tier: str = Field(min_length=1)
    language: str = Field(min_length=2)
    subject: str = Field(min_length=1)
    message: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class KbArticle(BaseModel):
    article_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: IssueType
    updated_at: str = Field(min_length=1)
    content: str = Field(min_length=1)


class RetrievedArticle(BaseModel):
    article_id: str = Field(min_length=1)
    score: float = Field(ge=0)
    title: str = Field(min_length=1)


class RetrievalResult(BaseModel):
    ticket_id: str = Field(min_length=1)
    retrieved_articles: list[RetrievedArticle]


class Classification(BaseModel):
    ticket_id: str = Field(min_length=1)
    issue_type: IssueType
    urgency: Urgency
    resolution_action: ResolutionAction
    customer_sensitive: bool
    requires_refund_caution: bool
    rationale: str = Field(min_length=1)


class DraftResponse(BaseModel):
    ticket_id: str = Field(min_length=1)
    draft_response: str = Field(min_length=1)
    citations: list[str]


class TriageDecision(BaseModel):
    ticket_id: str = Field(min_length=1)
    escalate: bool
    target_queue: TargetQueue
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_queue(self) -> "TriageDecision":
        if not self.escalate and self.target_queue != TargetQueue.NONE:
            raise ValueError("target_queue must be none when escalate is false")
        if self.escalate and self.target_queue == TargetQueue.NONE:
            raise ValueError("target_queue must not be none when escalate is true")
        return self


class LlmCallLog(BaseModel):
    stage: str = Field(pattern="^(classification|drafting)$")
    ticket_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    input_artifacts: list[str]
    output_artifact: str = Field(min_length=1)

