import pytest
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_switchboard.core.ports import LLMPort, LLMUnavailable


HA = "skill-homeassistant.oscillatelabsllc"
DATETIME = "ovos-skill-date-time.openvoiceos"
WEATHER = "skill-weather.neongeckocom"
DATA = "skill-data_controls.neongeckocom"

TEMPLATES = {
    (HA, "turn.on"): ["turn on (the|my|) {entity}", "turn (the|my|) {entity} on",
                      "(can|would) you turn on the {entity} please", "I would like the {entity} on"],
    (HA, "turn.off"): ["turn off (the|my|) {entity}", "turn (the|my|) {entity} off",
                       "(can|would) you turn off the {entity} please", "I would like the {entity} off"],
    (DATETIME, "what_time_is_it"): ["what time is it", "what time is it in {location}", "tell me the time",
                                    "I need to check the time"],
    (DATETIME, "what_day_is_it"): ["what day is it", "tell me the current day", "what is today's date"],
    (DATA, "clear_data"): ["clear my {dataset} data", "delete all my {dataset}"],
}
WEATHER_KEYWORD = {"skill_id": WEATHER, "intent_name": "HandleCurrentWeather", "lang": "en-us",
                   "required": [{"name": "Weather", "samples": ["weather", "forecast"]}],
                   "optional": [{"name": "Location", "samples": ["in seattle", "in (paris|london)"]}],
                   "one_of": [], "excluded": []}
DIALOG_LINES = ["I turned the lamp on for you", "Okay, the lights are off now", "It is currently {time}",
                "Sorry, I didn't catch that", "The forecast calls for rain later today"]


def register_all(bus):
    for (skill_id, name), samples in TEMPLATES.items():
        bus.emit(Message("ovos.intent.register.template",
                         {"skill_id": skill_id, "intent_name": name, "lang": "en-us", "samples": samples}))
    bus.emit(Message("ovos.intent.register.keyword", WEATHER_KEYWORD))


class StubLLM(LLMPort):
    """Stands in for LLMClient: scripted tool choice and gate probability."""

    def __init__(self, tool="ignore", args=None, p_yes=0.99, fail=False):
        self.tool, self.args, self.p_yes, self.fail = tool, args or {}, p_yes, fail
        self.calls = 0

    def route(self, utterance, tools):
        self.calls += 1
        if self.fail:
            raise LLMUnavailable("stub endpoint down")
        assert self.tool in {t["function"]["name"] for t in tools}, f"{self.tool} not shortlisted"
        return {"tool": self.tool, "args": self.args, "ms": 1.0}

    def p_addressed(self, utterance):
        return {"p_yes": self.p_yes, "ms": 1.0}


@pytest.fixture(name="bus")
def bus_fixture():
    return FakeBus()
