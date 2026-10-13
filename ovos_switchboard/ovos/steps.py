"""Switchboard's OVOS pipeline steps (OVOS-PIPELINE-1 plugins), one per stage, each placed independently in
`intents.pipeline`:

  ovos-switchboard-echo-pipeline-plugin        the assistant hearing itself -> decline
  ovos-switchboard-exact-pipeline-plugin       declared template phrases -> dispatch
  ovos-switchboard-classifier-pipeline-plugin  confident picks -> dispatch; decline class -> decline or pass on
  ovos-switchboard-llm-pipeline-plugin         LLM with manifest-derived tools and an addressee gate

A step turns its stage's verdict into OVOS terms: DISPATCH -> IntentHandlerMatch to the skill, DECLINE -> a match
for the Hangup skill (silent), PASS -> None. Each step has its own config section (`intents.<pipeline id>`) over the
shared `intents.switchboard` section. mode "shadow" logs verdicts and never matches.
"""
import threading
import time

from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, PipelinePlugin

from ovos_switchboard import data
from ovos_switchboard.adapters.bus_registry import session_of
from ovos_switchboard.adapters.openai_llm import LLMClient
from ovos_switchboard.core.classifier import Classifier
from ovos_switchboard.core.device import harvest_dialog_negatives
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.shortlist import Shortlist
from ovos_switchboard.core.stages import DECLINE_POLICIES, ClassifierStage, EchoStage, ExactStage, LLMStage
from ovos_switchboard.core.tools import build_tools
from ovos_switchboard.core.verdict import Outcome
from ovos_switchboard.ovos.runtime import LOG, MODES, SwitchboardRuntime

HANGUP_INTENT = "decline"  # interface contract with Hangup (HangupSkill.DECLINE_INTENT)
SHARED_SECTION = "switchboard"
SYNTHETIC_FILE = "synthetic_examples_slotted.jsonl"


def shared_config():
    try:
        from ovos_config import Configuration
        return Configuration().get("intents", {}).get(SHARED_SECTION, {})
    except ImportError:  # pragma: no cover
        return {}


class SwitchboardStep(PipelinePlugin):
    name = ""
    off_thread_in_shadow = False   # slow steps (the LLM) never block the pipeline in shadow mode

    def __init__(self, bus=None, config=None, runtime=None):
        super().__init__(bus, config)
        self.owns_runtime = runtime is None
        self.runtime = runtime or SwitchboardRuntime.acquire(self.bus, shared_config())
        self.mode = self.config.get("mode", self.runtime.settings.mode)
        if self.mode not in MODES:
            raise ValueError(f"{self.name}: mode must be one of {MODES}, got {self.mode!r}")
        self.setup()

    def setup(self):
        """Provide a stage builder to the runtime (steps with trained artifacts) or build the stage directly."""

    def current_stage(self):
        return self.runtime.stage(self.name)

    def shutdown(self):
        if self.owns_runtime:
            self.runtime.release()

    def match(self, utterances, lang, message):
        stage = self.current_stage()
        if not stage or not utterances or not self.runtime.serves(lang):
            return None
        utterance, session_id = utterances[0], session_of(message)
        record = self.runtime.record(message, utterance)
        if record.held or (self.mode == "shadow" and record.settled_by):
            return None
        context = self.runtime.context(utterance, session_id)
        if self.mode == "shadow":
            if self.off_thread_in_shadow:
                threading.Thread(target=self._shadow, args=(stage, utterance, session_id, lang, context, record),
                                 daemon=True, name=f"switchboard-{self.name}-shadow").start()
            else:
                self._shadow(stage, utterance, session_id, lang, context, record)
            return None
        verdict, ms = self._decide(stage, utterance, context)
        return self._act(verdict, utterance, session_id, lang, context, record, ms)

    @staticmethod
    def _decide(stage, utterance, context):
        t0 = time.perf_counter()
        verdict = stage.decide(utterance, context)
        return verdict, round((time.perf_counter() - t0) * 1000, 1)

    def _shadow(self, stage, utterance, session_id, lang, context, record):
        verdict, ms = self._decide(stage, utterance, context)
        if verdict.decisive and not record.settled_by:
            record.settled_by = self.name
        self._log(verdict, "shadow", utterance, session_id, lang, context, ms)

    def _act(self, verdict, utterance, session_id, lang, context, record, ms):
        if verdict.outcome is Outcome.PASS:
            self._log(verdict, "pass", utterance, session_id, lang, context, ms)
            return None
        if verdict.outcome is Outcome.DECLINE:
            self._log(verdict, "decline", utterance, session_id, lang, context, ms)
            return self._hangup(verdict, utterance)
        if self.runtime.settings.high_risk == "defer" and self.runtime.schema.risk(verdict.intent) == "high":
            record.held = True
            self._log(verdict, "held", utterance, session_id, lang, context, ms, "high-risk intent, no confirm flow")
            return None
        self._log(verdict, "dispatch", utterance, session_id, lang, context, ms)
        return IntentHandlerMatch(match_type=verdict.label, match_data=dict(verdict.match_data),
                                  skill_id=verdict.label.split(":", 1)[0], utterance=utterance)

    def _hangup(self, verdict, utterance):
        skill_id = self.runtime.settings.hangup_skill_id
        details = {k: verdict.details[k] for k in ("confidence", "p_addressed", "llm_tool", "echo_run")
                   if verdict.details.get(k) is not None}
        return IntentHandlerMatch(match_type=f"{skill_id}:{HANGUP_INTENT}", skill_id=skill_id, utterance=utterance,
                                  match_data={"reason": verdict.reason, "stage": verdict.stage, **details})

    def _log(self, verdict, action, utterance, session_id, lang, context, ms, reason=None):
        self.runtime.log.write({
            "event": "switchboard.verdict", "step": self.name, "mode": self.mode, "action": action,
            "outcome": verdict.outcome.value, "reason": reason or verdict.reason, "utterance": utterance,
            "session_id": session_id, "lang": lang, **context, "ms": ms, "intent": verdict.intent,
            "label": verdict.label, "match_data": verdict.match_data, **verdict.details})


class EchoStep(SwitchboardStep):
    name = "echo"

    def setup(self):
        self.stage = EchoStage(int(self.config.get("echo_run_min", 4)))

    def current_stage(self):
        return self.stage


class ExactStep(SwitchboardStep):
    name = "exact"

    def setup(self):
        self.runtime.provide(self.name, lambda b: ExactStage(ExactLayer(b.device, b.schema), b.resolver))


class ClassifierStep(SwitchboardStep):
    name = "classifier"

    def setup(self):
        c = self.config
        self.threshold = float(c.get("threshold", 0.5))
        self.policy = c.get("route_decline_class", "active")
        self.active_s = float(c.get("active_s", 120))
        self.dialog_negatives = bool(c.get("dialog_negatives", True))
        if self.policy not in DECLINE_POLICIES:
            raise ValueError(f"route_decline_class must be one of {DECLINE_POLICIES}, got {self.policy!r}")
        self.synthetic = data.jsonl(SYNTHETIC_FILE)
        self.runtime.provide(self.name, self.build)

    def build(self, b):
        rows = list(self.synthetic)
        if self.dialog_negatives:
            rows += harvest_dialog_negatives(self.runtime.dialog_lines(b.lang), b.definitions, b.lang)
        return ClassifierStage(Classifier(b.device, b.schema, rows), b.resolver, self.threshold, self.policy,
                               self.active_s)


class LLMStep(SwitchboardStep):
    name = "llm"
    off_thread_in_shadow = True

    def setup(self):
        c = self.config
        if not c.get("url"):
            # OVOS instantiates every installed pipeline plugin, listed in the pipeline or not: stay inert.
            LOG.warning("ovos-switchboard-llm-pipeline-plugin has no `url` configured; the LLM step never matches")
            return
        self.llm = LLMClient(c["url"], c.get("model", ""), data.prompt("fallback_strict.txt"),
                             data.prompt("addressee_gate.txt"), float(c.get("timeout_s", 4)), c.get("api_key"))
        self.gate, self.k = float(c.get("gate", 0.9)), int(c.get("shortlist_k", 12))
        self.runtime.provide(self.name, self.build)

    def build(self, b):
        tools = build_tools(b.definitions, b.lang)
        return LLMStage(self.llm, Shortlist(tools, b.device.examples), tools, b.schema, b.resolver, self.gate, self.k)
