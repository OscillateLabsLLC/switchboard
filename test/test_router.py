import pytest

from conftest import DATA, DIALOG_LINES, HA, WEATHER, StubLLM, register_all
from ovos_switchboard.device import Device, bundled_jsonl, harvest_dialog_negatives
from ovos_switchboard.registry import IntentRegistry
from ovos_switchboard.router import Router, RouterConfig
from ovos_switchboard.schema import Schema
from ovos_switchboard.tools import build_tools


@pytest.fixture(scope="module")
def definitions():
    from ovos_utils.fakebus import FakeBus
    bus = FakeBus()
    reg = IntentRegistry(bus)
    register_all(bus)
    return reg.definitions()


def make_router(definitions, llm=None, **config):
    synthetic = bundled_jsonl("synthetic_examples_slotted.jsonl")
    synthetic += harvest_dialog_negatives(DIALOG_LINES, definitions, "en-us")
    device = Device.from_definitions(definitions, synthetic)
    return Router(device, Schema.load(), build_tools(definitions), llm, RouterConfig(**config))


@pytest.fixture(scope="module")
def router(definitions):
    return make_router(definitions)


def test_exact_phrase_routes_with_native_slots(router):
    d = router.route("turn on the desk lamp")
    assert (d["stage"], d["intent"], d["label"]) == ("exact", "device.on", f"{HA}:turn.on")
    assert d["match_data"] == {"entity": "desk lamp"}


def test_echo_guard_declines_before_anything_else(router):
    d = router.route("turn on the desk lamp", {"echo_run": 4})
    assert (d["stage"], d["intent"]) == ("echo", None)


def test_paraphrase_goes_to_the_classifier(router):
    d = router.route("whats the time right now")
    assert d["stage"] == "classifier"
    assert d["intent"] == "time.current"


def test_assistant_speech_is_the_decline_class(router):
    d = router.route("i turned the lamp on for you")
    assert d["intent"] is None and d["decline_class"]
    assert "needs_llm" not in d  # no LLM configured


def test_decline_class_reaches_llm_only_when_assistant_is_active(definitions):
    r = make_router(definitions, llm=StubLLM())
    assert not r.route_fast("i turned the lamp on for you", {"gap_s": None}).get("needs_llm")
    assert r.route_fast("i turned the lamp on for you", {"gap_s": 30}).get("needs_llm")
    always = make_router(definitions, llm=StubLLM(), route_decline_class="always")
    assert always.route_fast("i turned the lamp on for you", {"gap_s": None}).get("needs_llm")


def test_llm_skill_tool_dispatches_with_tool_args(definitions):
    r = make_router(definitions, llm=StubLLM("homeassistant__turn_off", {"entity": "porch light"}))
    d = r.route_llm("kill the porch light", {"intent": None, "stage": "classifier", "needs_llm": True})
    assert (d["stage"], d["intent"], d["label"]) == ("llm", "device.off", f"{HA}:turn.off")
    assert d["match_data"] == {"entity": "porch light"}


def test_keyword_tool_gets_vocab_match_data(definitions):
    r = make_router(definitions, llm=StubLLM("weather__HandleCurrentWeather"))
    d = r.route_llm("is the forecast any good in paris", {"intent": None, "stage": "classifier"})
    assert d["label"] == f"{WEATHER}:HandleCurrentWeather"
    assert d["match_data"] == {"Weather": "forecast", "Location": "in paris"}


def test_failed_addressee_gate_declines(definitions):
    r = make_router(definitions, llm=StubLLM("homeassistant__turn_off", p_yes=0.2))
    d = r.route_llm("turn that thing off will you", {"intent": None, "stage": "classifier"})
    assert (d["stage"], d["intent"]) == ("llm_gate", None) and "label" not in d


def test_meta_tools_carry_no_label(definitions):
    r = make_router(definitions, llm=StubLLM("ask_knowledge", {"question": "why is the sky blue"}))
    d = r.route_llm("why is the sky blue", {"intent": None, "stage": "classifier"})
    assert d["intent"] == "knowledge.ask" and "label" not in d


def test_high_risk_intent_is_visible_to_the_caller(router):
    d = router.route("clear my browsing data")
    assert d["label"] == f"{DATA}:clear_data"
    assert Schema.load().risk(d["intent"]) == "high"


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        RouterConfig(route_decline_class="sometimes")
