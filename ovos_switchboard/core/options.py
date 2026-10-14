"""Tunable options, each defined once with its measured default and its validation.

The OVOS steps read them from mycroft.conf with `options_from`; the harness composes them in RouterConfig. Field
names are the config keys.
"""
from dataclasses import MISSING, dataclass, fields


DECLINE_POLICIES = ("never", "active", "always")  # when decline-class rows pass on to later stages (the LLM)


def options_from(cls, config):
    """An options dataclass from the config keys it declares. Unknown keys are ignored; numbers are coerced."""
    config = config or {}
    return cls(**{f.name: _coerce(f.default, config[f.name]) for f in fields(cls) if f.name in config})


def _coerce(default, value):
    if default is MISSING or default is None or isinstance(default, bool):
        return value
    return type(default)(value) if isinstance(default, (int, float)) else value


@dataclass(frozen=True)
class SpeechOptions:
    """What the runtime remembers of the assistant's own speech, for echo and "assistant active" context."""
    echo_window_s: float = 60.0     # echo compares against speech this recent
    speech_history_s: float = 600.0
    speech_lines_max: int = 64      # per session


@dataclass(frozen=True)
class EchoOptions:
    """When an utterance counts as the assistant hearing itself."""
    echo_run_min: int = 4           # contiguous words shared with recent assistant speech


@dataclass(frozen=True)
class ExactOptions:
    """When a catch-all template match ('play {query}', 'call {contact}') gives way to the classifier."""
    # Confidence in a DIFFERENT intent (or the decline class) that overrules a catch-all match; 0 never overrules.
    # Chosen on dev, confirmed on test (intent-eval exp_guard_rules.py, 2026-10-13).
    catch_all_yield: float = 0.8


@dataclass(frozen=True)
class PrefixOptions:
    """Trailing speech after a declared phrase."""
    prefix_min_literal: int = 3     # literal words a slot-free template needs to match as a prefix


@dataclass(frozen=True)
class ClassifierOptions:
    """Dispatch confidence, and what the decline class does."""
    threshold: float = 0.5
    route_decline_class: str = "active"
    active_s: float = 120.0         # the "assistant active" window for route_decline_class="active"
    dialog_negatives: bool = True   # train the decline class on installed skills' .dialog lines

    def __post_init__(self):
        if self.route_decline_class not in DECLINE_POLICIES:
            raise ValueError(f"route_decline_class must be one of {DECLINE_POLICIES}, "
                             f"got {self.route_decline_class!r}")


@dataclass(frozen=True)
class LLMOptions:
    """The addressee gate and the tool shortlist size."""
    gate: float = 0.9               # P(addressed) needed to act
    shortlist_k: int = 12           # skill tools offered per utterance
