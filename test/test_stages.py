import pytest
from ovos_bus_client.message import Message

from conftest import DATA, DATETIME, DIALOG_LINES, HA, WEATHER, StubLLM, register_all
from ovos_switchboard import data
from ovos_switchboard.adapters.bus_registry import IntentRegistry
from ovos_switchboard.core.classifier import Classifier
from ovos_switchboard.core.device import Device, harvest_dialog_negatives
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.labels import LabelSelector
from ovos_switchboard.core.options import ClassifierOptions, EchoOptions, LLMOptions
from ovos_switchboard.core.router import Router
from ovos_switchboard.core.schema import Schema
from ovos_switchboard.core.shortlist import Shortlist
from ovos_switchboard.core.slots import SlotExtractor
from ovos_switchboard.core.stages import ClassifierStage, EchoStage, ExactStage, LabelResolver, LLMStage, PrefixStage
from ovos_switchboard.core.tools import build_tools
from ovos_switchboard.core.verdict import Outcome, Verdict


COLD = {"gap_s": None, "echo_run": 0}


class Parts:
    def __init__(self):
        from ovos_utils.fakebus import FakeBus
        bus = FakeBus()
        reg = IntentRegistry(bus)
        register_all(bus)
        self.definitions = reg.definitions()
        self.device = Device.from_definitions(self.definitions)
        self.schema = Schema.load()
        self.resolver = LabelResolver(LabelSelector(self.device, self.schema), SlotExtractor(self.device, self.schema))
        self.tools = build_tools(self.definitions)
        self.synthetic = data.jsonl("synthetic_examples_slotted.jsonl") + harvest_dialog_negatives(
            DIALOG_LINES, self.definitions, "en-us")
        self.classifier = Classifier.for_device(self.device, self.schema, self.synthetic)

    def classifier_stage(self, **kw):
        return ClassifierStage(self.classifier, self.resolver, ClassifierOptions(**kw))

    def llm_stage(self, llm, **kw):
        return LLMStage(llm, Shortlist(self.tools, self.device.examples), self.schema, self.resolver, LLMOptions(**kw))


@pytest.fixture(name="parts", scope="module")
def parts_fixture():
    return Parts()


def test_echo_stage():
    stage = EchoStage(EchoOptions(echo_run_min=4))
    assert stage.decide("x", {"echo_run": 4}).outcome is Outcome.DECLINE
    assert stage.decide("x", {"echo_run": 3}).outcome is Outcome.PASS


def test_exact_stage_dispatches_declared_phrases_with_native_slots(parts):
    stage = ExactStage(ExactLayer(parts.device, parts.schema), parts.resolver)
    v = stage.decide("turn on the desk lamp")
    assert (v.outcome, v.intent, v.label, v.match_data) == (Outcome.DISPATCH, "device.on", f"{HA}:turn.on",
                                                            {"entity": "desk lamp"})
    assert stage.decide("is it going to be cold tonight").outcome is Outcome.PASS


def test_exact_stage_needs_no_training(parts):
    """The exact step can run alone: built from templates only, no classifier, no LLM."""
    stage = ExactStage(ExactLayer(parts.device, parts.schema), parts.resolver)
    assert stage.decide("clear my browsing data").label == f"{DATA}:clear_data"


def test_classifier_dispatches_paraphrases(parts):
    v = parts.classifier_stage().decide("whats the time right now", COLD)
    assert (v.outcome, v.intent) == (Outcome.DISPATCH, "time.current")
    assert v.details["confidence"] >= 0.5


@pytest.mark.parametrize("policy, gap, outcome", [
    ("active", None, Outcome.DECLINE), ("active", 30, Outcome.PASS), ("active", 300, Outcome.DECLINE),
    ("always", None, Outcome.PASS), ("never", 30, Outcome.DECLINE),
])
def test_decline_class_policy(parts, policy, gap, outcome):
    v = parts.classifier_stage(route_decline_class=policy).decide("i turned the lamp on for you", {"gap_s": gap})
    assert v.outcome is outcome and v.details["decline_class"]


def test_low_confidence_passes_on(parts):
    v = parts.classifier_stage(threshold=0.999).decide("whats the time right now", COLD)
    assert (v.outcome, v.reason) == (Outcome.PASS, "low confidence")


def test_invalid_policy_is_rejected(parts):
    with pytest.raises(ValueError):
        parts.classifier_stage(route_decline_class="sometimes")


def test_llm_skill_tool_dispatches_with_tool_args(parts):
    v = parts.llm_stage(StubLLM("homeassistant__turn_off", {"entity": "porch light"})).decide("kill the porch light")
    assert (v.outcome, v.intent, v.label, v.match_data) == (Outcome.DISPATCH, "device.off", f"{HA}:turn.off",
                                                            {"entity": "porch light"})


def test_llm_keyword_tool_gets_vocab_match_data(parts):
    v = parts.llm_stage(StubLLM("weather__HandleCurrentWeather")).decide("is the forecast any good in paris")
    assert v.label == f"{WEATHER}:HandleCurrentWeather"
    assert v.match_data == {"Weather": "forecast", "Location": "in paris"}


@pytest.mark.parametrize("llm, outcome, reason", [
    (StubLLM("homeassistant__turn_off", p_yes=0.2), Outcome.DECLINE, "addressee_gate"),
    (StubLLM("ignore"), Outcome.DECLINE, "ignore"),
    (StubLLM(fail=True), Outcome.DECLINE, "llm_unavailable"),
    (StubLLM("easter_eggs_joke", {"query": "algo"}, invented=True), Outcome.DECLINE, "unknown_tool"),
    (StubLLM("ask_knowledge", {"question": "why is the sky blue"}), Outcome.PASS, "llm chose ask_knowledge"),
])
def test_llm_outcomes(parts, llm, outcome, reason):
    v = parts.llm_stage(llm).decide("why is the sky blue")
    assert (v.outcome, v.reason) == (outcome, reason)


def test_a_dispatch_must_name_a_skill_label():
    with pytest.raises(ValueError):
        Verdict("exact", Outcome.DISPATCH, intent="device.on")


def test_router_composes_stages_in_measured_order(parts):
    r = Router(parts.device, parts.schema, parts.tools, synthetic=parts.synthetic, llm=StubLLM("ignore"))
    assert r.route("turn on the desk lamp", {"echo_run": 4}) == {"intent": None, "stage": "echo"}
    assert r.route("turn on the desk lamp")["stage"] == "exact"
    assert r.route("whats the time right now")["stage"] == "classifier"
    assert r.route("i turned the lamp on for you", {"gap_s": 10})["stage"] == "llm"


def test_router_declines_an_invented_tool(parts):
    r = Router(parts.device, parts.schema, parts.tools, synthetic=parts.synthetic,
               llm=StubLLM("easter_eggs_joke", invented=True))
    d = r.route("i turned the lamp on for you", {"gap_s": 10})
    assert (d["intent"], d["stage"], d["llm_tool"]) == (None, "llm", "easter_eggs_joke")


def test_classifier_state_round_trips_exactly(parts):
    restored = Classifier.from_state(*parts.classifier.state())
    for utterance in ("whats the time right now", "i turned the lamp on for you", "turn off the porch light"):
        assert restored.rank(utterance) == parts.classifier.rank(utterance)


class RankedClassifier:
    def __init__(self, ranked):
        self.ranked = ranked

    def rank(self, _utterance):
        return self.ranked


def test_unmapped_top_class_passes_with_its_own_reason(parts):
    stage = ClassifierStage(RankedClassifier([("__unmapped__:skill-x.someone:do_thing", 0.97)]), parts.resolver)
    assert stage.decide("do the thing", COLD).reason == "unmapped intent"


def picked(parts, intent, utterance):
    """The classifier stage's verdict when the model ranks `intent` first."""
    return ClassifierStage(RankedClassifier([(intent, 0.97)]), parts.resolver).decide(utterance, COLD)


def test_slot_required_label_with_nothing_filled_passes(parts):
    """#37: every turn.on sample needs {entity}; a knowledge question gave the skill nothing to act on."""
    v = picked(parts, "device.on", "what is the boiling point of mercury in kelvin")
    assert (v.outcome, v.reason) == (Outcome.PASS, "required slot unfilled")
    # no intent, so the router routes onward; the decision log keeps the label
    assert (v.intent, v.label, v.details["unfilled"]) == (None, None, f"{HA}:turn.on")


def test_gated_pass_keeps_the_classifier_details(parts):
    """The router copies these into the decision record for a classifier PASS."""
    v = picked(parts, "device.on", "what is the boiling point of mercury in kelvin")
    assert (v.details["confidence"], v.details["label_how"]) == (0.97, "only label")


def test_required_slot_gate_is_classifier_only(parts, monkeypatch):
    """Exact and prefix matches dispatch as before even for a label the gate would hold."""
    monkeypatch.setattr(parts.resolver, "slot_required", lambda label: True)
    layer = ExactLayer(parts.device, parts.schema)
    v = ExactStage(layer, parts.resolver).decide("what time is it")
    assert (v.outcome, v.match_data) == (Outcome.DISPATCH, {})
    layer.enable_prefix(3)
    v = PrefixStage(layer, parts.resolver).decide("tell me the time oh and also")
    assert (v.outcome, v.label) == (Outcome.DISPATCH, f"{DATETIME}:what_time_is_it")


def slot_required_after(*registrations):
    """LabelSelector.slot_required for (skill_id, intent_name, lang, samples) registrations, on an en-us device."""
    from ovos_utils.fakebus import FakeBus
    bus = FakeBus()
    reg = IntentRegistry(bus)
    for skill_id, name, lang, samples in registrations:
        bus.emit(Message("ovos.intent.register.template",
                         {"skill_id": skill_id, "intent_name": name, "lang": lang, "samples": samples}))
    return LabelSelector(Device.from_definitions(reg.definitions(), "en-us"), Schema.load()).slot_required


def test_a_typed_sample_that_can_omit_its_slot_keeps_the_label_ungated():
    assert f"{HA}:turn.on" not in slot_required_after(
        (HA, "turn.on", "en-us", ["turn on [the {entity:device}]", "turn on the {entity}"]))


def test_typed_samples_count_as_needing_their_slot():
    assert f"{HA}:turn.on" in slot_required_after((HA, "turn.on", "en-us", ["turn on the {entity:device}"]))


def test_another_languages_samples_do_not_gate_the_device_language():
    assert f"{DATETIME}:what_time_is_it" not in slot_required_after(
        (DATETIME, "what_time_is_it", "en-us", ["what time is it", "what time is it in {location}"]),
        (DATETIME, "what_time_is_it", "de-de", ["wie spät ist es in {ort}"]))


def test_slot_required_label_with_the_slot_filled_dispatches(parts):
    v = picked(parts, "device.on", "hey could you turn on the desk lamp")
    assert (v.outcome, v.label, v.match_data) == (Outcome.DISPATCH, f"{HA}:turn.on", {"entity": "desk lamp"})


def test_label_with_a_slot_free_sample_dispatches_without_data(parts):
    """what_time_is_it has "what time is it": an empty slot is a valid reading of it."""
    v = picked(parts, "time.current", "whats the time right now")
    assert (v.outcome, v.match_data) == (Outcome.DISPATCH, {})


def test_adapt_label_dispatches_without_data(parts):
    """Keyword intents are not templates: the gate does not apply to them."""
    v = picked(parts, "weather.current", "is it going to be cold tonight")
    assert (v.outcome, v.label) == (Outcome.DISPATCH, f"{WEATHER}:HandleCurrentWeather")
