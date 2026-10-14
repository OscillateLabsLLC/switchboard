import pytest
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_switchboard.adapters.bus_registry import IntentRegistry
from ovos_switchboard.core.device import Device
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.labels import LabelSelector
from ovos_switchboard.core.options import ExactOptions
from ovos_switchboard.core.schema import Schema
from ovos_switchboard.core.slots import SlotExtractor
from ovos_switchboard.core.stages import ExactStage, LabelResolver
from ovos_switchboard.core.templates import is_catch_all
from ovos_switchboard.core.verdict import Outcome


COMMS = "skill-communication.neongeckocom"
HA = "skill-homeassistant.oscillatelabsllc"
TEMPLATES = {(COMMS, "call"): ["call {contact}"],
             (HA, "turn.on"): ["turn on (the|my|) {entity}"]}


class Parts:
    def __init__(self):
        bus = FakeBus()
        reg = IntentRegistry(bus)
        for (skill_id, name), samples in TEMPLATES.items():
            bus.emit(Message("ovos.intent.register.template",
                             {"skill_id": skill_id, "intent_name": name, "lang": "en-us", "samples": samples}))
        self.device = Device.from_definitions(reg.definitions())
        self.schema = Schema.load()
        self.resolver = LabelResolver(LabelSelector(self.device, self.schema), SlotExtractor(self.device, self.schema))
        self.call = self.schema.canon_or_unmapped(f"{COMMS}:call")

    def stage(self, ranking=None, threshold=0.8):
        def ranker(_utterance):
            return ranking
        return ExactStage(ExactLayer(self.device, self.schema), self.resolver, ranker=ranker,
                          options=ExactOptions(catch_all_yield=threshold))


@pytest.fixture(name="parts", scope="module")
def parts_fixture():
    return Parts()


@pytest.mark.parametrize("line, catch_all", [
    ("call {contact}", True), ("play {query}", True), ("timer [for] {duration}", True),
    ("turn on (the|my|) {entity}", False), ("stop", False), ("what time is it", False),
])
def test_catch_all_is_a_slot_behind_one_literal_word(line, catch_all):
    assert is_catch_all(line) is catch_all


def test_catch_all_dispatches_when_the_classifier_agrees_or_is_unsure(parts):
    utt = "call mom"
    for ranking in (None, [(parts.call, 0.99)], [("assistant.about", 0.79)]):
        v = parts.stage(ranking).decide(utt)
        assert (v.outcome, v.label) == (Outcome.DISPATCH, f"{COMMS}:call")


def test_confident_classifier_overrules_a_catch_all(parts):
    """Neon's own self-introduction must not place a call (intent-eval exact_audit.py)."""
    utt = "call me neon. i am an intelligent personal assistant"
    v = parts.stage([("assistant.about", 0.93)]).decide(utt)
    assert v.outcome is Outcome.PASS
    assert v.details == {"catch_all": parts.call, "overruled_by": "assistant.about", "confidence": 0.93}
    assert parts.stage([("__decline__", 0.85)]).decide(utt).outcome is Outcome.PASS


def test_zero_threshold_never_overrules(parts):
    v = parts.stage([("assistant.about", 0.99)], threshold=0).decide("call mom")
    assert v.outcome is Outcome.DISPATCH


def test_templates_with_two_literal_words_never_consult_the_classifier(parts):
    def ranker(_utterance):
        raise AssertionError("a non-catch-all match consulted the classifier")
    stage = ExactStage(ExactLayer(parts.device, parts.schema), parts.resolver, ranker=ranker)
    assert stage.decide("turn on the desk lamp").outcome is Outcome.DISPATCH


def test_match_reports_catch_all_only_matches(parts):
    layer = ExactLayer(parts.device, parts.schema)
    assert layer.match("call mom") == (parts.call, True)
    assert layer.match("turn on the lamp") == ("device.on", False)
    assert layer.match("what is the weather") == (None, False)
