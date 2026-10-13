import json
import time

import pytest
from ovos_bus_client.message import Message

from conftest import DIALOG_LINES, HA, StubLLM, register_all
from ovos_switchboard.core.verdict import Outcome, Verdict
from ovos_switchboard.ovos.runtime import Settings, SwitchboardRuntime
from ovos_switchboard.ovos.steps import ClassifierStep, EchoStep, ExactStep, LLMStep


HANGUP = "ovos-skill-hangup.oscillatelabsllc:decline"
LLM_CONFIG = {"url": "http://127.0.0.1:1/v1/chat/completions", "model": "m"}


def utterance_message():
    """A fresh Message per utterance round, as ovos-core sends."""
    return Message("recognizer_loop:utterance", {}, {"session": {"session_id": "default"}})


@pytest.fixture(name="runtime")
def runtime_fixture(bus, tmp_path):
    rt = SwitchboardRuntime(bus, Settings(mode="active", retrain_debounce_s=3600,
                                          decision_log_path=str(tmp_path / "decisions.jsonl")),
                            dialog_lines=lambda lang: DIALOG_LINES, start=False)
    register_all(bus)
    yield rt
    rt.shutdown()


def records(tmp_path, event="switchboard.verdict"):
    path = tmp_path / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [r for r in rows if r["event"] == event]


def run_pipeline(steps, utterance, lang="en-us"):
    """What ovos-core does: try each step in order with the same message; the first match wins."""
    msg = utterance_message()
    for step in steps:
        match = step.match([utterance], lang, msg)
        if match:
            return step.name, match
    return None, None


class ScriptedStage:
    def __init__(self, verdict):
        self.verdict = verdict

    def decide(self, _utterance, _context):
        return self.verdict


def test_exact_step_works_alone(runtime):
    exact = ExactStep(runtime.bus, {}, runtime=runtime)
    runtime.rebuild()
    assert set(runtime.stages) == {"exact"}, "only what an installed step asked for is built"
    name, match = run_pipeline([exact], "turn on the desk lamp")
    assert (name, match.match_type, match.match_data) == ("exact", f"{HA}:turn.on", {"entity": "desk lamp"})


def test_classifier_step_works_alone(runtime):
    classifier = ClassifierStep(runtime.bus, {}, runtime=runtime)
    runtime.rebuild()
    name, match = run_pipeline([classifier], "whats the time right now")
    assert name == "classifier" and match.match_type.endswith(":what_time_is_it")


def test_echo_step_needs_no_registrations(bus):
    rt = SwitchboardRuntime(bus, Settings(mode="active", decision_log=False), start=False)
    echo = EchoStep(bus, {}, runtime=rt)  # measured setting: 4+ contiguous words
    bus.emit(Message("speak", {"utterance": "the porch light is now off"}, utterance_message().context))
    _, match = run_pipeline([echo], "porch light is now off")
    assert (match.match_type, match.match_data["reason"]) == (HANGUP, "echo")
    assert run_pipeline([echo], "turn off the porch light") == (None, None)


def test_full_pipeline_declines_assistant_speech_at_the_classifier(runtime):
    steps = [EchoStep(runtime.bus, {}, runtime=runtime), ExactStep(runtime.bus, {}, runtime=runtime),
             ClassifierStep(runtime.bus, {}, runtime=runtime)]
    runtime.rebuild()
    name, match = run_pipeline(steps, "i turned the lamp on for you")
    assert (name, match.match_type, match.match_data["reason"]) == ("classifier", HANGUP, "decline_class")


def test_high_risk_hold_binds_later_steps(runtime, tmp_path):
    exact, classifier = ExactStep(runtime.bus, {}, runtime=runtime), ClassifierStep(runtime.bus, {}, runtime=runtime)
    runtime.rebuild()
    assert run_pipeline([exact, classifier], "clear my browsing data") == (None, None)
    held = [r for r in records(tmp_path) if r["action"] == "held"]
    assert [r["step"] for r in held] == ["exact"], "the classifier never got to re-decide the held utterance"


def test_llm_step_dispatches_declines_and_passes(runtime):
    llm = LLMStep(runtime.bus, LLM_CONFIG, runtime=runtime)
    runtime.rebuild()
    stage = runtime.stage("llm")
    stage.llm = StubLLM("homeassistant__turn_off", {"entity": "fan"})
    assert run_pipeline([llm], "kill the fan")[1].match_type == f"{HA}:turn.off"
    stage.llm = StubLLM(fail=True)
    assert run_pipeline([llm], "kill the fan")[1].match_data["reason"] == "llm_unavailable"
    stage.llm = StubLLM("ask_knowledge", {"question": "why is the sky blue"})
    assert run_pipeline([llm], "why is the sky blue") == (None, None)


def test_llm_step_without_url_is_inert(runtime):
    llm = LLMStep(runtime.bus, {}, runtime=runtime)
    runtime.rebuild()
    assert "llm" not in runtime.stages
    assert run_pipeline([llm], "kill the fan") == (None, None)


def test_shadow_mode_never_matches_and_later_steps_skip_settled_utterances(bus, tmp_path):
    rt = SwitchboardRuntime(bus, Settings(mode="shadow", retrain_debounce_s=3600,
                                          decision_log_path=str(tmp_path / "decisions.jsonl")),
                            dialog_lines=lambda lang: DIALOG_LINES, start=False)
    register_all(bus)
    steps = [ExactStep(bus, {}, runtime=rt), ClassifierStep(bus, {}, runtime=rt)]
    rt.rebuild()
    assert run_pipeline(steps, "turn on the desk lamp") == (None, None)
    assert [(r["step"], r["outcome"]) for r in records(tmp_path)] == [("exact", "dispatch")]
    run_pipeline(steps, "whats the time right now")
    assert [r["step"] for r in records(tmp_path)][1:] == ["exact", "classifier"]


def test_shadow_llm_runs_off_the_pipeline_thread(runtime, tmp_path):
    llm = LLMStep(runtime.bus, {**LLM_CONFIG, "mode": "shadow"}, runtime=runtime)
    runtime.rebuild()
    runtime.stage("llm").llm = StubLLM("ignore")
    assert run_pipeline([llm], "so anyway") == (None, None)
    deadline = time.time() + 5
    while time.time() < deadline and not records(tmp_path):
        time.sleep(0.02)
    assert (records(tmp_path)[0]["mode"], records(tmp_path)[0]["reason"]) == ("shadow", "ignore")


def test_per_step_mode_overrides_shared(runtime):
    assert ExactStep(runtime.bus, {"mode": "shadow"}, runtime=runtime).mode == "shadow"
    with pytest.raises(ValueError):
        ExactStep(runtime.bus, {"mode": "loud"}, runtime=runtime)


def test_other_languages_are_left_alone(runtime):
    echo = EchoStep(runtime.bus, {}, runtime=runtime)
    echo.stage = ScriptedStage(Verdict("echo", Outcome.DECLINE, reason="echo"))
    assert echo.match(["hallo"], "de-de", utterance_message()) is None
    assert echo.match(["hello"], "en-US", utterance_message()) is not None


def test_ovos_outcomes_are_logged(runtime, tmp_path):
    runtime.bus.emit(Message("ovos.intent.matched", {"utterance": "what time is it", "intent_name": "x:y",
                                                     "pipeline_id": "ovos-padatious-pipeline-plugin"}))
    assert records(tmp_path, "ovos.intent.matched")[0]["pipeline_id"] == "ovos-padatious-pipeline-plugin"


def test_runtime_is_shared_per_bus_and_released(bus):
    a = SwitchboardRuntime.acquire(bus, {"decision_log": False})
    b = SwitchboardRuntime.acquire(bus, {"decision_log": False})
    assert a is b
    a.release()
    b.release()
    assert SwitchboardRuntime.acquire(bus, {"decision_log": False}) is not a


def test_invalid_settings_fail_at_load(bus):
    with pytest.raises(ValueError):
        Settings.from_config({"mode": "loud"})
    rt = SwitchboardRuntime(bus, Settings(decision_log=False), start=False)
    with pytest.raises(ValueError):
        ClassifierStep(bus, {"route_decline_class": "sometimes"}, runtime=rt)
