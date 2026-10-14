import json
import time

import pytest
from ovos_bus_client.message import Message

from conftest import DIALOG_LINES, HA, StubLLM, register_all
from ovos_switchboard.core.verdict import Outcome, Verdict
from ovos_switchboard.ovos.runtime import Settings, SwitchboardRuntime
from ovos_switchboard.ovos.steps import ClassifierStep, DeciderStep, EchoStep, ExactStep, LLMStep, PrefixStep


HANGUP = "ovos-skill-hangup.oscillatelabsllc:decline"
DOUBLECHECK = "ovos-skill-doublecheck.oscillatelabsllc"
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


def test_doublechecks_quote_of_the_user_is_not_echo(bus):
    """Live trial: "what time is it" again after "I think you said: what time is it..." was declined as echo."""
    rt = SwitchboardRuntime(bus, Settings(mode="active", decision_log=False), start=False)
    echo = EchoStep(bus, {}, runtime=rt)
    question = "I think you said: what time is it. Should I do that?"
    context = {**utterance_message().context, "skill_id": DOUBLECHECK}
    bus.emit(Message("speak", {"utterance": question}, context))
    assert run_pipeline([echo], "what time is it") == (None, None)
    assert rt.context("what time is it", "default")["gap_s"] is not None, "it still counts as the assistant speaking"
    bus.emit(Message("speak", {"utterance": question}, utterance_message().context))
    assert run_pipeline([echo], "what time is it")[1].match_data["reason"] == "echo"


def test_full_pipeline_declines_assistant_speech_at_the_classifier(runtime):
    steps = [EchoStep(runtime.bus, {}, runtime=runtime), ExactStep(runtime.bus, {}, runtime=runtime),
             ClassifierStep(runtime.bus, {}, runtime=runtime)]
    runtime.rebuild()
    name, match = run_pipeline(steps, "i turned the lamp on for you")
    assert (name, match.match_type, match.match_data["reason"]) == ("classifier", HANGUP, "decline_class")


class StockStep:
    """Stands in for a stock matcher after Switchboard (Adapt, Padatious) that would run anything it sees."""
    name = "stock"

    def __init__(self):
        self.seen = []

    def match(self, utterances, _lang, _message):
        self.seen.append(utterances[0])
        return "stock match"


def test_high_risk_hold_consumes_the_utterance(runtime, tmp_path):
    """A held intent goes to Hangup, silently: returning no match would let a later stock step run it anyway."""
    exact, classifier = ExactStep(runtime.bus, {}, runtime=runtime), ClassifierStep(runtime.bus, {}, runtime=runtime)
    stock = StockStep()
    runtime.rebuild()
    name, match = run_pipeline([exact, classifier, stock], "clear my browsing data")
    assert (name, match.match_type) == ("exact", HANGUP)
    assert match.match_data["reason"] == "held_high_risk"
    assert (match.match_data["held_intent"], match.match_data["held_label"]) == (
        "data.clear", "skill-data_controls.neongeckocom:clear_data")
    assert stock.seen == [], "nothing after Switchboard saw the held utterance"
    held = [r for r in records(tmp_path) if r["action"] == "held"]
    assert [(r["step"], r["reason"]) for r in held] == [("exact", "held_high_risk")]


@pytest.fixture(name="confirming")
def confirming_fixture(bus, tmp_path):
    rt = SwitchboardRuntime(bus, Settings(mode="active", high_risk="confirm", retrain_debounce_s=3600,
                                          decision_log_path=str(tmp_path / "decisions.jsonl")),
                            dialog_lines=lambda lang: DIALOG_LINES, start=False)
    register_all(bus)
    yield rt
    rt.shutdown()


def hold_unlock(rt):
    exact, stock = ExactStep(rt.bus, {}, runtime=rt), StockStep()
    rt.rebuild()
    rt.stages["exact"] = ScriptedStage(Verdict("exact", Outcome.DISPATCH, "lock.unlock", f"{HA}:unlock",
                                               {"entity": "front door"}, {}))
    name, match = run_pipeline([exact, stock], "unlock the front door")
    assert name == "exact" and stock.seen == [], "nothing after Switchboard saw the held utterance"
    return match


@pytest.mark.parametrize("loaded", [True, None])  # None: ovos-core didn't say which skills are loaded
def test_confirm_hands_the_held_intent_to_doublecheck(confirming, tmp_path, loaded):
    """high_risk "confirm": Doublecheck gets what it needs to replay the dispatch; nothing later runs it."""
    confirming.confirm_loaded = loaded
    match = hold_unlock(confirming)
    assert (match.match_type, match.skill_id) == (f"{DOUBLECHECK}:confirm", DOUBLECHECK)
    assert match.match_data == {"held_intent": "lock.unlock", "held_label": f"{HA}:unlock",
                                "held_match_data": {"entity": "front door"}}
    assert [r["action"] for r in records(tmp_path)] == ["held"]


def test_confirm_without_doublecheck_holds_silently(confirming):
    """Live trial: a dispatch to a missing skill was silent but left ovos-core waiting out its 5-minute timeout."""
    confirming.confirm_loaded = False
    match = hold_unlock(confirming)
    assert (match.match_type, match.match_data["reason"]) == (HANGUP, "held_high_risk")


@pytest.mark.parametrize("skills, loaded", [([DOUBLECHECK], True), ([], False)])
def test_startup_check_records_whether_doublecheck_is_loaded(confirming, skills, loaded):
    confirming.bus.on("ovos.skills.list", lambda m: confirming.bus.emit(
        m.response({"ok": True, "skills": [{"skill_id": s} for s in skills]})))
    confirming._check_skills()  # pylint: disable=protected-access
    assert confirming.confirm_loaded is loaded


def test_unknown_high_risk_policy_is_rejected():
    with pytest.raises(ValueError):
        Settings(high_risk="confrim")


def test_ocp_verdicts_are_delegated_to_ocps_own_step(runtime, tmp_path):
    """OCP's ocp:play handler needs match data only OCP's matcher adds (live trial: KeyError 'media_type')."""
    exact, classifier = ExactStep(runtime.bus, {}, runtime=runtime), ClassifierStep(runtime.bus, {}, runtime=runtime)
    ocp = StockStep()
    runtime.rebuild()
    runtime.stages["exact"] = ScriptedStage(Verdict("exact", Outcome.DISPATCH, "media.play", "ocp:play",
                                                    {"query": "groove salad"}, {}))
    assert run_pipeline([exact, classifier, ocp], "play groove salad") == ("stock", "stock match")
    assert ocp.seen == ["play groove salad"]
    rows = [(r["step"], r["action"]) for r in records(tmp_path)]
    assert rows == [("exact", "delegate")], "the classifier never re-decided the delegated utterance"


def test_first_build_skips_the_debounce_and_answers_readiness(bus, tmp_path):
    """Live: "I am ready" came 8 s before Switchboard was; 5 s of it was the debounce after skills had loaded."""
    rt = SwitchboardRuntime(bus, Settings(mode="active", retrain_debounce_s=3600, bus_timeout_s=0.1,
                                          decision_log_path=str(tmp_path / "decisions.jsonl")),
                            dialog_lines=lambda lang: DIALOG_LINES, start=False)
    register_all(bus)
    ExactStep(bus, {}, runtime=rt)
    replies = []
    bus.on("mycroft.switchboard.is_ready.response", lambda m: replies.append(m.data["status"]))
    bus.emit(Message("mycroft.switchboard.is_ready"))
    rt._skills_loaded()  # pylint: disable=protected-access
    deadline = time.monotonic() + 10
    while not rt.ready() and time.monotonic() < deadline:
        time.sleep(0.05)
    bus.emit(Message("mycroft.switchboard.is_ready"))
    rt.shutdown()
    assert replies == [False, True], "built well inside the 3600 s debounce, and only then ready"


def unsupported_after(runtime, classifier_verdict, utterance="can you pass me the salt"):
    """Run classifier -> llm -> stock with scripted verdicts; the LLM finds no tool."""
    classifier, llm = ClassifierStep(runtime.bus, {}, runtime=runtime), LLMStep(runtime.bus, LLM_CONFIG,
                                                                                runtime=runtime)
    stock = StockStep()
    runtime.rebuild()
    runtime.stages["classifier"] = ScriptedStage(classifier_verdict)
    runtime.stages["llm"] = ScriptedStage(Verdict("llm", Outcome.PASS, reason="llm chose unsupported",
                                                  details={"llm_tool": "unsupported", "p_addressed": 1.0}))
    return run_pipeline([classifier, llm, stock], utterance), stock


def test_not_for_me_plus_no_tool_declines_instead_of_reaching_stock_steps(runtime):
    """Live warm-context trial: "can you pass me the salt" leaked to stock m2v (missed_alerts)."""
    (name, match), stock = unsupported_after(runtime, Verdict(
        "classifier", Outcome.PASS, reason="decline class, assistant active", details={"decline_class": True}))
    assert (name, match.match_type, match.match_data["reason"]) == ("llm", HANGUP, "decline_class_unsupported")
    assert stock.seen == []


def test_an_unsupported_request_the_classifier_was_unsure_about_still_passes(runtime):
    (name, _), stock = unsupported_after(runtime, Verdict("classifier", Outcome.PASS, reason="low confidence"),
                                         "book me a flight to denver")
    assert name == "stock" and stock.seen == ["book me a flight to denver"]


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


def test_decider_step_without_its_extra_is_inert(runtime, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    decider = DeciderStep(runtime.bus, {}, runtime=runtime)
    runtime.rebuild()
    assert "decider" not in runtime.stages
    assert run_pipeline([decider], "kill the fan") == (None, None)


def test_decider_step_dispatches_and_declines(runtime, monkeypatch):
    from test_decider import StubDecider
    monkeypatch.setattr("importlib.util.find_spec", lambda name: True)  # the "decider" extra, simulated
    monkeypatch.setattr("ovos_switchboard.adapters.decider_model.TasksourceDecider",
                        lambda spec: StubDecider("homeassistant__turn_off"))
    decider = DeciderStep(runtime.bus, {}, runtime=runtime)
    runtime.rebuild()
    assert run_pipeline([decider], "turn off the fan")[1].match_type == f"{HA}:turn.off"
    runtime.stage("decider").decider = StubDecider("ignore")
    match = run_pipeline([decider], "pass me the salt")[1]
    assert match.skill_id == "ovos-skill-hangup.oscillatelabsllc"
    assert match.match_data["reason"] == "decider: none of these"


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


def wait_for(condition, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not condition():
        time.sleep(0.02)
    return condition()


def test_rebuilds_wait_until_core_has_loaded_skills(bus):
    rt = SwitchboardRuntime(bus, Settings(retrain_debounce_s=0.01, bus_timeout_s=0.1, decision_log=False),
                            dialog_lines=lambda lang: DIALOG_LINES, start=False)
    ExactStep(bus, {}, runtime=rt)
    register_all(bus)
    time.sleep(0.2)
    assert rt.stage("exact") is None, "no rebuild while skills are still registering"
    bus.emit(Message("mycroft.skills.initialized"))
    assert wait_for(lambda: rt.stage("exact") is not None)
    rt.shutdown()


def test_a_reloaded_pipeline_builds_when_core_is_already_ready(bus):
    bus.on("mycroft.skills.is_ready", lambda m: bus.emit(m.response({"status": True})))
    rt = SwitchboardRuntime(bus, Settings(retrain_debounce_s=0.01, bus_timeout_s=0.1, decision_log=False),
                            dialog_lines=lambda lang: DIALOG_LINES)
    ExactStep(bus, {}, runtime=rt)
    register_all(bus)
    assert wait_for(lambda: rt.stage("exact") is not None)
    rt.shutdown()


def test_wake_phrases_reach_the_exact_and_prefix_steps(bus):
    rt = SwitchboardRuntime(bus, Settings(mode="active", retrain_debounce_s=3600, decision_log=False,
                                          wake_phrases=("hey_mycroft",)), start=False)
    exact, prefix = ExactStep(bus, {}, runtime=rt), PrefixStep(bus, {}, runtime=rt)
    register_all(bus)
    rt.rebuild()
    assert run_pipeline([exact], "Hey Mycroft, turn on the desk lamp")[1].match_type == f"{HA}:turn.on"
    name, match = run_pipeline([exact, prefix], "hey mycroft tell me the time oh and")
    assert (name, match.match_type) == ("prefix", "ovos-skill-date-time.openvoiceos:what_time_is_it")
    rt.shutdown()
