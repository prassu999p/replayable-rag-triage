import pytest

from rag_triage.schemas import PipelineStage
from rag_triage.stage_machine import PipelineStateMachine


def test_stage_machine_accepts_required_stage_order() -> None:
    machine = PipelineStateMachine()

    for stage in [
        PipelineStage.INPUTS_LOADED,
        PipelineStage.KB_INDEXED,
        PipelineStage.TICKETS_NORMALISED,
        PipelineStage.EVIDENCE_RETRIEVED,
        PipelineStage.TICKETS_CLASSIFIED,
        PipelineStage.RESPONSES_DRAFTED,
        PipelineStage.ESCALATIONS_DECIDED,
        PipelineStage.OUTPUTS_VALIDATED,
        PipelineStage.RESULTS_FINALISED,
    ]:
        machine.advance(stage)

    assert machine.current == PipelineStage.RESULTS_FINALISED
    assert machine.history == [
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


def test_stage_machine_rejects_out_of_order_stage() -> None:
    machine = PipelineStateMachine()

    with pytest.raises(ValueError, match="Cannot advance"):
        machine.advance(PipelineStage.RESPONSES_DRAFTED)

