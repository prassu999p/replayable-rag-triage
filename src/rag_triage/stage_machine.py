from dataclasses import dataclass, field

from rag_triage.schemas import PipelineStage


STAGE_ORDER = [
    PipelineStage.INIT,
    PipelineStage.INPUTS_LOADED,
    PipelineStage.KB_INDEXED,
    PipelineStage.TICKETS_NORMALISED,
    PipelineStage.EVIDENCE_RETRIEVED,
    PipelineStage.TICKETS_CLASSIFIED,
    PipelineStage.RESPONSES_DRAFTED,
    PipelineStage.ESCALATIONS_DECIDED,
    PipelineStage.OUTPUTS_VALIDATED,
    PipelineStage.RESULTS_FINALISED,
]


@dataclass
class PipelineStateMachine:
    current: PipelineStage = PipelineStage.INIT
    history: list[PipelineStage] = field(default_factory=lambda: [PipelineStage.INIT])

    def advance(self, next_stage: PipelineStage) -> None:
        expected_stage = STAGE_ORDER[STAGE_ORDER.index(self.current) + 1]
        if next_stage != expected_stage:
            raise ValueError(
                f"Cannot advance from {self.current.value} to {next_stage.value}; "
                f"expected {expected_stage.value}"
            )
        self.current = next_stage
        self.history.append(next_stage)

