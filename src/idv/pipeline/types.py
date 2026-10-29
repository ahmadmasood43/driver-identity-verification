from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class Decision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REVIEW = "review"


@dataclass(frozen=True)
class PipelineResult:
    decision: Decision
    scores: dict[str, float]
    reason_codes: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)


class Pipeline(Protocol):
    def run(self, selfie: bytes, id_front: bytes) -> PipelineResult: ...
