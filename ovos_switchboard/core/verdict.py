"""A stage's answer for one utterance."""
from dataclasses import dataclass, field
from enum import Enum


class Outcome(Enum):
    DISPATCH = "dispatch"  # run this skill intent
    DECLINE = "decline"    # not meant for the assistant: stay silent
    PASS = "pass"          # not this stage's call: later pipeline steps decide


@dataclass(frozen=True)
class Verdict:
    stage: str
    outcome: Outcome
    intent: str | None = None    # canonical intent (kept on PASS too, e.g. knowledge.ask, for the record)
    label: str | None = None     # skill_id:intent_name to dispatch
    match_data: dict = field(default_factory=dict)  # what that skill reads: its own slot / vocab names
    slots: dict = field(default_factory=dict)       # canonical slot names, for the record
    reason: str | None = None
    details: dict = field(default_factory=dict)     # confidence, top3, llm_tool, p_addressed, ...

    def __post_init__(self):
        if self.outcome is Outcome.DISPATCH and not (self.label and ":" in self.label):
            raise ValueError(f"{self.stage}: a dispatch needs a skill_id:intent_name label, got {self.label!r}")

    @property
    def decisive(self):
        return self.outcome is not Outcome.PASS


def passed(stage, reason, **kw):
    return Verdict(stage, Outcome.PASS, reason=reason, **kw)


def declined(stage, reason, **kw):
    return Verdict(stage, Outcome.DECLINE, reason=reason, **kw)
