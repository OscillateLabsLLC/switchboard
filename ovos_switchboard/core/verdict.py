"""A stage's answer for one utterance."""
from dataclasses import dataclass, field
from enum import Enum


class Outcome(Enum):
    """What a stage decided for an utterance."""
    DISPATCH = "dispatch"  # run this skill intent
    DECLINE = "decline"    # not meant for the assistant: stay silent
    PASS = "pass"          # not this stage's call: later pipeline steps decide


@dataclass(frozen=True)
class Verdict:  # pylint: disable=too-many-instance-attributes  # a value object
    """One stage's decision for one utterance, with what the dispatch needs and why."""
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
        """Whether the stage settled the utterance (dispatch or decline)."""
        return self.outcome is not Outcome.PASS


def passed(stage, reason, **kw):
    """A PASS verdict."""
    return Verdict(stage, Outcome.PASS, reason=reason, **kw)


def declined(stage, reason, **kw):
    """A DECLINE verdict."""
    return Verdict(stage, Outcome.DECLINE, reason=reason, **kw)
