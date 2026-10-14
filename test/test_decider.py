import pytest

from conftest import HA, WEATHER
from ovos_switchboard.core.options import DeciderOptions
from ovos_switchboard.core.ports import DeciderPort
from ovos_switchboard.core.router import Router, RouterConfig
from ovos_switchboard.core.shortlist import Shortlist
from ovos_switchboard.core.stages import DeciderStage
from ovos_switchboard.core.tools import DECIDER_META, DECIDER_NONE, DECIDER_QUESTION, plain_option
from ovos_switchboard.core.verdict import Outcome
from test_stages import COLD, Parts


class StubDecider(DeciderPort):
    """Scripted choice and probability; records what it was offered."""

    def __init__(self, choice, p=0.9):
        self.choice, self.p, self.offered, self.question = choice, p, None, None

    def choose(self, state, question, options):
        self.offered, self.question = options, question
        assert self.choice in options, f"{self.choice} not offered"
        return {"choice": self.choice, "probabilities": {k: (self.p if k == self.choice else 0.0) for k in options},
                "ms": 1.0}


@pytest.fixture(name="parts", scope="module")
def parts_fixture():
    return Parts()


def stage(parts, decider, **kw):
    return DeciderStage(decider, Shortlist(parts.tools, parts.device.examples), parts.schema, parts.resolver,
                        DeciderOptions(**kw))


def test_plain_option_is_the_first_sentence_and_two_examples_with_slots_named():
    tool = {"function": {"name": "homeassistant__turn_off", "description": (
        'Turn off (homeassistant skill). Example requests: "turn off the <entity>"; "switch off <entity>"; '
        '"kill the <entity>"')}}
    assert plain_option(tool) == "Turn off (homeassistant skill): turn off the entity; switch off entity"


def test_offered_options_are_plain_with_none_of_these_last(parts):
    d = StubDecider("homeassistant__turn_off")
    stage(parts, d).decide("turn off the porch light")
    assert d.question == DECIDER_QUESTION
    assert list(d.offered)[-1] == DECIDER_NONE and d.offered[DECIDER_NONE] == DECIDER_META[DECIDER_NONE]
    assert "unsupported" not in d.offered
    assert d.offered["ask_knowledge"] == DECIDER_META["ask_knowledge"]


def test_a_confident_skill_pick_dispatches_with_switchboards_own_match_data(parts):
    v = stage(parts, StubDecider("weather__HandleCurrentWeather")).decide("is the forecast any good in paris")
    assert (v.outcome, v.label) == (Outcome.DISPATCH, f"{WEATHER}:HandleCurrentWeather")
    assert v.match_data == {"Weather": "forecast", "Location": "in paris"}
    assert v.details["decider_choice"] == "weather__HandleCurrentWeather"


@pytest.mark.parametrize("decider, kw, outcome, reason", [
    (StubDecider(DECIDER_NONE), {}, Outcome.DECLINE, "decider: none of these"),
    (StubDecider("homeassistant__turn_off", p=0.5), {}, Outcome.DECLINE, "decider: below threshold"),
    (StubDecider("homeassistant__turn_off", p=0.5), {"on_unsure": "pass"}, Outcome.PASS, "decider: below threshold"),
    (StubDecider("homeassistant__turn_off", p=0.5), {"threshold": 0.4}, Outcome.DISPATCH, None),
])
def test_unsure_outcomes(parts, decider, kw, outcome, reason):
    v = stage(parts, decider, **kw).decide("turn off the porch light")
    assert (v.outcome, v.reason) == (outcome, reason)


def test_a_meta_pick_passes_with_its_canonical_intent(parts):
    v = stage(parts, StubDecider("ask_knowledge")).decide("why is the sky blue")
    assert (v.outcome, v.intent) == (Outcome.PASS, "knowledge.ask")


def test_invalid_unsure_policy_is_rejected():
    with pytest.raises(ValueError):
        DeciderOptions(on_unsure="maybe")


def test_router_runs_the_decider_after_the_classifier(parts):
    r = Router(parts.device, parts.schema, parts.tools, synthetic=parts.synthetic,
               decider=StubDecider("homeassistant__turn_off"), config=RouterConfig())
    d = r.route("i turned the lamp on for you", {**COLD, "gap_s": 10})
    assert (d["stage"], d["label"], d["decider_choice"]) == ("decider", f"{HA}:turn.off", "homeassistant__turn_off")
    assert r.route("turn on the desk lamp")["stage"] == "exact"


def test_block_mask_follows_the_trained_layout():
    torch = pytest.importorskip("torch")
    from ovos_switchboard.adapters.decider_model import OPTION, QUESTION, STATE, block_mask
    kinds = torch.tensor([[STATE, STATE, QUESTION, OPTION, -1]])
    allow = block_mask(kinds, torch.arange(5)[None], None)[0, 0]
    assert allow[0].tolist() == [True, True, False, False, False]   # state sees only state
    assert allow[2].tolist() == [True, True, True, False, False]    # question sees state and question
    assert allow[3].tolist() == [True, True, True, True, False]     # option sees all three
    assert allow[4].tolist() == [False, False, False, False, True]  # padding sees only itself
    assert not block_mask(kinds, torch.arange(5)[None], 1)[0, 0, 3, 0]  # outside the local window


def test_option_text_is_key_and_description():
    from ovos_switchboard.adapters.decider_model import option_text
    assert option_text("stop", "stop what the assistant is doing") == "stop: stop what the assistant is doing"
    assert option_text("stop", "") == "stop"
