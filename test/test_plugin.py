import json
import time

import pytest
from ovos_bus_client.message import Message

from conftest import DIALOG_LINES, HA, register_all
from ovos_switchboard.llm import LLMClient, LLMUnavailable
from ovos_switchboard.plugin import HANGUP_SKILL_ID, Settings, SwitchboardPipeline

MSG = Message("recognizer_loop:utterance", {}, {"session": {"session_id": "default"}})


def make_plugin(bus, tmp_path, **config):
    p = SwitchboardPipeline(bus, {"mode": "active", "decision_log_path": str(tmp_path / "decisions.jsonl"),
                                  "retrain_debounce_s": 60, **config})
    p.trainer.dialog_lines = lambda lang: DIALOG_LINES
    register_all(bus)
    p.trainer.rebuild()
    return p


def records(tmp_path, event="switchboard.decision"):
    path = tmp_path / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [r for r in rows if r["event"] == event]


class ScriptedRouter:
    """Fixed decisions, so each tier's handling is tested without depending on what the classifier learns."""

    def __init__(self, fast, llm=None):
        self.fast, self.llm = fast, llm

    def route_fast(self, utterance, context):
        return dict(self.fast)

    def route_llm(self, utterance, decision):
        if isinstance(self.llm, Exception):
            raise self.llm
        return dict(self.llm)


def scripted(bus, tmp_path, fast, llm=None, **config):
    p = SwitchboardPipeline(bus, {"mode": "active", "decision_log_path": str(tmp_path / "decisions.jsonl"), **config})
    p.trainer.router = ScriptedRouter(fast, llm)
    return p


def tiers(p, utterance, lang="en-us"):
    return [m([utterance], lang, MSG) for m in (p.match_high, p.match_medium, p.match_low)]


def test_exact_phrase_dispatches_at_high_tier(bus, tmp_path):
    p = make_plugin(bus, tmp_path)
    match = p.match_high(["turn on the desk lamp"], "en-us", MSG)
    assert (match.match_type, match.skill_id) == (f"{HA}:turn.on", HA)
    assert match.match_data == {"entity": "desk lamp"}
    assert records(tmp_path)[-1]["action"] == "dispatch"


def test_echo_of_own_speech_goes_to_hangup(bus, tmp_path):
    p = make_plugin(bus, tmp_path)
    bus.emit(Message("speak", {"utterance": "turn on the desk lamp is what you asked"}, MSG.context))
    match = p.match_high(["turn on the desk lamp is what you asked"], "en-us", MSG)
    assert match.match_type == f"{HANGUP_SKILL_ID}:decline"
    assert match.match_data["reason"] == "echo"


def test_high_risk_intent_is_never_acted_on(bus, tmp_path):
    p = make_plugin(bus, tmp_path)
    assert tiers(p, "clear my browsing data") == [None, None, None]
    assert records(tmp_path)[-1]["reason"] == "high-risk intent, no confirm flow"


def test_shadow_mode_logs_and_never_matches(bus, tmp_path):
    p = make_plugin(bus, tmp_path, mode="shadow")
    assert tiers(p, "turn on the desk lamp") == [None, None, None]
    shadow = records(tmp_path)
    assert len(shadow) == 1, "all three tiers share one decision"
    assert (shadow[0]["action"], shadow[0]["stage"], shadow[0]["label"]) == ("shadow", "exact", f"{HA}:turn.on")
    assert shadow[0]["fast_ms"] >= 0 and "gap_s" in shadow[0]


def test_shadow_mode_runs_the_llm_off_the_pipeline_thread(bus, tmp_path):
    p = scripted(bus, tmp_path, {"intent": None, "stage": "classifier", "needs_llm": True},
                 {"intent": None, "stage": "llm_gate", "llm_tool": "ignore", "p_addressed": 0.1}, mode="shadow")
    assert p.match_high(["so anyway"], "en-us", MSG) is None
    deadline = time.time() + 5
    while time.time() < deadline and not records(tmp_path):
        time.sleep(0.02)
    assert records(tmp_path)[0]["stage"] == "llm_gate"


def test_decline_class_goes_to_hangup_at_medium_tier(bus, tmp_path):
    p = scripted(bus, tmp_path, {"intent": None, "stage": "classifier", "decline_class": True, "confidence": 0.8})
    high, medium, _ = tiers(p, "we should leave by six")
    assert high is None
    assert medium.match_type == f"{HANGUP_SKILL_ID}:decline"
    assert medium.match_data == {"reason": "decline_class", "stage": "classifier", "confidence": 0.8}


def test_llm_skill_choice_dispatches_at_low_tier(bus, tmp_path):
    p = scripted(bus, tmp_path, {"intent": None, "stage": "classifier", "needs_llm": True},
                 {"intent": "device.off", "stage": "llm", "label": f"{HA}:turn.off", "match_data": {"entity": "fan"}})
    high, medium, low = tiers(p, "kill the fan")
    assert high is None and medium is None
    assert (low.match_type, low.match_data) == (f"{HA}:turn.off", {"entity": "fan"})


def test_knowledge_questions_fall_through_to_common_query(bus, tmp_path):
    p = scripted(bus, tmp_path, {"intent": None, "stage": "classifier", "needs_llm": True},
                 {"intent": "knowledge.ask", "stage": "llm", "llm_tool": "ask_knowledge"})
    assert tiers(p, "why is the sky blue") == [None, None, None]
    assert records(tmp_path)[-1]["action"] == "defer"


@pytest.mark.parametrize("llm, reason", [
    ({"intent": None, "stage": "llm", "llm_tool": "ignore"}, "ignore"),
    ({"intent": None, "stage": "llm_gate", "llm_tool": "homeassistant__turn_off", "p_addressed": 0.3}, "addressee_gate"),
    (LLMUnavailable("connection refused"), "llm_unavailable"),
])
def test_llm_declines_go_to_hangup(bus, tmp_path, llm, reason):
    p = scripted(bus, tmp_path, {"intent": None, "stage": "classifier", "needs_llm": True}, llm)
    match = p.match_low(["mm hmm sure"], "en-us", MSG)
    assert match.match_type == f"{HANGUP_SKILL_ID}:decline"
    assert match.match_data["reason"] == reason


def test_other_languages_are_left_to_other_pipelines(bus, tmp_path):
    p = scripted(bus, tmp_path, {"intent": "device.on", "stage": "exact", "label": f"{HA}:turn.on"})
    assert p.match_high(["schalte die lampe ein"], "de-de", MSG) is None
    assert p.match_high(["turn on the lamp"], "en-US", MSG) is not None


def test_no_router_yet_means_no_match(bus, tmp_path):
    p = SwitchboardPipeline(bus, {"mode": "active", "decision_log": False})
    assert p.match_high(["turn on the lamp"], "en-us", MSG) is None


def test_ovos_outcomes_are_logged_for_comparison(bus, tmp_path):
    scripted(bus, tmp_path, {"intent": None, "stage": "classifier"})
    bus.emit(Message("ovos.intent.matched", {"utterance": "what time is it", "skill_id": "x", "intent_name": "x:y",
                                             "pipeline_id": "ovos-padatious-pipeline-plugin"}))
    assert records(tmp_path, "ovos.intent.matched")[0]["pipeline_id"] == "ovos-padatious-pipeline-plugin"


def test_settings():
    s = Settings.from_config({"mode": "active", "threshold": 0.6, "route_decline_class": "always", "unknown": 1})
    assert (s.mode, s.router.threshold, s.router.route_decline_class) == ("active", 0.6, "always")
    with pytest.raises(ValueError):
        Settings.from_config({"mode": "loud"})


def test_llm_client_is_built_only_when_configured(bus, tmp_path):
    p = SwitchboardPipeline(bus, {"decision_log": False, "llm": {"url": "http://localhost:1/v1/chat/completions",
                                                                 "model": "m", "timeout_s": 2}})
    assert isinstance(p.trainer.llm, LLMClient) and p.trainer.llm.timeout_s == 2.0
    assert SwitchboardPipeline(bus, {"decision_log": False}).trainer.llm is None
