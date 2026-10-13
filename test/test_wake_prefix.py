import pytest

from conftest import DATETIME, HA
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.stages import ExactStage, PrefixStage
from ovos_switchboard.core.verdict import Outcome
from ovos_switchboard.core.wake import strip_wake_phrase, wake_phrase_pattern
from test_stages import parts_fixture  # noqa: F401  (registers the "parts" fixture here)


WAKE = wake_phrase_pattern(["hey_mycroft", "hey neon"])


@pytest.mark.parametrize("utterance, expected", [
    ("Hey Mycroft, where were you born?", "where were you born?"),
    ("hey, mycroft where were you born", "where were you born"),
    ("Hey Neon, what time is it?", "what time is it?"),
    ("Hey Mycroft.", "Hey Mycroft."),                          # a bare wake phrase stays for later stages
    ("tell hey mycroft hello", "tell hey mycroft hello"),      # only a LEADING phrase is stripped
    ("hey mycroftian", "hey mycroftian"),                      # whole words only
])
def test_strip_wake_phrase(utterance, expected):
    assert strip_wake_phrase(utterance, WAKE) == expected


def test_no_wake_phrases_strips_nothing():
    assert wake_phrase_pattern([]) is None
    assert strip_wake_phrase("Hey Mycroft, what time is it", None) == "Hey Mycroft, what time is it"


def test_exact_matches_past_a_wake_phrase(parts):
    stage = ExactStage(ExactLayer(parts.device, parts.schema), parts.resolver, WAKE)
    v = stage.decide("Hey Mycroft, turn on the desk lamp")
    assert (v.outcome, v.label, v.match_data) == (Outcome.DISPATCH, f"{HA}:turn.on", {"entity": "desk lamp"})
    without = ExactStage(ExactLayer(parts.device, parts.schema), parts.resolver)
    assert without.decide("Hey Mycroft, turn on the desk lamp").outcome is Outcome.PASS


def test_prefix_dispatches_trailing_speech_after_a_declared_phrase(parts):
    layer = ExactLayer(parts.device, parts.schema)
    layer.enable_prefix(3)
    stage = PrefixStage(layer, parts.resolver, WAKE)
    assert stage.decide("hey mycroft tell me the time oh and also").label == f"{DATETIME}:what_time_is_it"
    assert stage.decide("what time is it").outcome is Outcome.PASS, "an exact phrase is not trailing speech"
    assert stage.decide("turn on the lamp and the fan").outcome is Outcome.PASS, "slot templates never prefix"
