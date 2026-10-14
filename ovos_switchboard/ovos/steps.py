"""Switchboard's OVOS pipeline steps (OVOS-PIPELINE-1 plugins), one per stage, each placed independently in
`intents.pipeline`:

  ovos-switchboard-echo-pipeline-plugin        the assistant hearing itself -> decline
  ovos-switchboard-exact-pipeline-plugin       declared template phrases -> dispatch
  ovos-switchboard-classifier-pipeline-plugin  confident picks -> dispatch; decline class -> decline or pass on
  ovos-switchboard-prefix-pipeline-plugin      trailing speech after a declared phrase -> dispatch
  ovos-switchboard-llm-pipeline-plugin         LLM with manifest-derived tools and an addressee gate
  ovos-switchboard-decider-pipeline-plugin     a local typed-decision model picks a shortlisted intent (no LLM host)

A step turns its stage's verdict into OVOS terms: DISPATCH -> IntentHandlerMatch to the skill, DECLINE -> a match
for the Hangup skill (silent), PASS -> None. A high-risk DISPATCH is held: Hangup again, or with high_risk "confirm"
a match for the Doublecheck skill, which asks first. Each step has its own config section (`intents.<pipeline id>`)
over the shared `intents.switchboard` section. mode "shadow" logs verdicts and never matches.
"""
import threading
import time
from dataclasses import dataclass

from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, PipelinePlugin

from ovos_switchboard import data
from ovos_switchboard.adapters.bus_registry import session_of
from ovos_switchboard.adapters.isolated_training import train_isolated
from ovos_switchboard.adapters.openai_llm import LLMClient, LLMEndpoint
from ovos_switchboard.core.classifier import Classifier, model_fingerprint, training_data
from ovos_switchboard.core.device import harvest_dialog_negatives
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.options import (
    ClassifierOptions,
    DeciderOptions,
    EchoOptions,
    ExactOptions,
    LLMOptions,
    PrefixOptions,
    options_from,
)
from ovos_switchboard.core.shortlist import Shortlist
from ovos_switchboard.core.stages import ClassifierStage, DeciderStage, EchoStage, ExactStage, LLMStage, PrefixStage
from ovos_switchboard.core.tools import build_tools
from ovos_switchboard.core.verdict import Outcome
from ovos_switchboard.ovos.runtime import LOG, MODES, SwitchboardRuntime, UtteranceRecord


HANGUP_INTENT = "decline"  # interface contract with Hangup (HangupSkill.DECLINE_INTENT)
SHARED_SECTION = "switchboard"
SYNTHETIC_FILE = "synthetic_examples_slotted.jsonl"
HANGUP_DETAIL_KEYS = ("confidence", "p_addressed", "llm_tool", "echo_run", "decider_choice", "decider_p")
OVERHEARD_REASON = "decline_class_unsupported"
HELD_REASON = "held_high_risk"
# Interface contract with Doublecheck (DoublecheckSkill.CONFIRM_INTENT, HELD_KEYS): it asks, and on "yes" replays
# held_label with held_match_data merged into the original message data, as ovos-core would have dispatched it.
CONFIRM_INTENT = "confirm"


def shared_config():
    """The `intents.switchboard` section of mycroft.conf."""
    try:
        from ovos_config import Configuration
        return Configuration().get("intents", {}).get(SHARED_SECTION, {})
    except ImportError:  # pragma: no cover
        return {}


@dataclass(frozen=True)
class Turn:
    """One utterance as a step sees it."""
    utterance: str
    session_id: str
    lang: str
    context: dict
    record: UtteranceRecord


class SwitchboardStep(PipelinePlugin):
    """One stage as an OVOS pipeline plugin: verdicts become matches, or None."""
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
        """This step's built stage, or None before the first build."""
        return self.runtime.stage(self.name)

    def shutdown(self):
        """Release the shared runtime if this step acquired it."""
        if self.owns_runtime:
            self.runtime.release()

    def match(self, utterances, lang, message):
        stage = self.current_stage()
        if not stage or not utterances or not self.runtime.serves(lang):
            return None
        utterance, session_id = utterances[0], session_of(message)
        record = self.runtime.record(message, utterance)
        if record.held or record.delegated or (self.mode == "shadow" and record.settled_by):
            return None
        turn = Turn(utterance, session_id, lang, self.runtime.context(utterance, session_id), record)
        if self.mode == "shadow":
            if self.off_thread_in_shadow:
                threading.Thread(target=self._shadow, args=(stage, turn), daemon=True,
                                 name=f"switchboard-{self.name}-shadow").start()
            else:
                self._shadow(stage, turn)
            return None
        verdict, ms = self._decide(stage, turn)
        return self._act(verdict, turn, ms)

    @staticmethod
    def _decide(stage, turn):
        t0 = time.perf_counter()
        verdict = stage.decide(turn.utterance, turn.context)
        return verdict, round((time.perf_counter() - t0) * 1000, 1)

    def _shadow(self, stage, turn):
        verdict, ms = self._decide(stage, turn)
        if verdict.decisive and not turn.record.settled_by:
            turn.record.settled_by = self.name
        self._log(verdict, "shadow", turn, ms)

    def _act(self, verdict, turn, ms):
        if verdict.outcome is Outcome.PASS:
            if verdict.details.get("decline_class"):
                turn.record.decline_class_passed = True
            if turn.record.decline_class_passed and verdict.details.get("llm_tool") == "unsupported":
                # "Not for me" got an exception only because the assistant had just spoken, and the LLM found
                # nothing to do: passing on handed overheard speech to whatever stock step came next (live: m2v).
                self._log(verdict, "decline", turn, ms, OVERHEARD_REASON)
                return self._hangup(verdict, turn.utterance, OVERHEARD_REASON)
            self._log(verdict, "pass", turn, ms)
            return None
        if verdict.outcome is Outcome.DECLINE:
            self._log(verdict, "decline", turn, ms)
            return self._hangup(verdict, turn.utterance)
        if self.runtime.settings.high_risk != "allow" and self.runtime.schema.risk(verdict.intent) == "high":
            return self._hold(verdict, turn, ms)
        skill_id = verdict.label.split(":", 1)[0]
        if skill_id in self.runtime.settings.delegate_skills:
            # Its own pipeline step (ocp-high, placed after the classifier) re-matches it with the data it needs.
            turn.record.delegated = True
            self._log(verdict, "delegate", turn, ms, f"delegated to {skill_id}'s own pipeline step")
            return None
        self._log(verdict, "dispatch", turn, ms)
        return IntentHandlerMatch(match_type=verdict.label, match_data=dict(verdict.match_data),
                                  skill_id=skill_id, utterance=turn.utterance)

    def _hangup(self, verdict, utterance, reason=None, extra=None):  # pylint: disable=R0913
        skill_id = self.runtime.settings.hangup_skill_id
        details = {k: verdict.details[k] for k in HANGUP_DETAIL_KEYS if verdict.details.get(k) is not None}
        return IntentHandlerMatch(match_type=f"{skill_id}:{HANGUP_INTENT}", skill_id=skill_id, utterance=utterance,
                                  match_data={"reason": reason or verdict.reason, "stage": verdict.stage, **details,
                                              **(extra or {})})

    def _hold(self, verdict, turn, ms):
        """Consumed, not passed: returning None would let a later stock step (Adapt, Padatious) run it anyway."""
        turn.record.held = True
        self._log(verdict, "held", turn, ms, HELD_REASON)
        held = {"held_intent": verdict.intent, "held_label": verdict.label}
        if self.runtime.settings.high_risk != "confirm":
            return self._hangup(verdict, turn.utterance, HELD_REASON, held)
        skill_id = self.runtime.settings.confirm_skill_id
        return IntentHandlerMatch(match_type=f"{skill_id}:{CONFIRM_INTENT}", skill_id=skill_id,
                                  utterance=turn.utterance,
                                  match_data={**held, "held_match_data": dict(verdict.match_data)})

    def _log(self, verdict, action, turn, ms, reason=None):
        self.runtime.log.write({
            "event": "switchboard.verdict", "step": self.name, "mode": self.mode, "action": action,
            "outcome": verdict.outcome.value, "reason": reason or verdict.reason, "utterance": turn.utterance,
            "session_id": turn.session_id, "lang": turn.lang, **turn.context, "ms": ms, "intent": verdict.intent,
            "label": verdict.label, "match_data": verdict.match_data, **verdict.details})


class EchoStep(SwitchboardStep):
    """Pipeline step: decline the assistant hearing itself."""
    name = "echo"

    def setup(self):
        self.stage = EchoStage(options_from(EchoOptions, self.config))

    def current_stage(self):
        return self.stage


class ExactStep(SwitchboardStep):
    """Pipeline step: dispatch declared template phrases."""
    name = "exact"

    def setup(self):
        self.options = options_from(ExactOptions, self.config)
        self.runtime.provide(self.name, self.build)

    def build(self, b):
        """The exact stage for the current registrations."""
        return ExactStage(ExactLayer(b.device, b.schema, options=self.options), b.resolver, b.wake,
                          ranker=self.rank, options=self.options)

    def rank(self, utterance):
        """The classifier's ranking, or None while it trains or when the classifier step isn't installed."""
        classifier = self.runtime.stage("classifier")
        return classifier.classifier.rank(utterance) if classifier else None


class PrefixStep(SwitchboardStep):
    """Pipeline step: dispatch trailing speech after a declared phrase. Place it after the classifier step."""
    name = "prefix"

    def setup(self):
        self.options = options_from(PrefixOptions, self.config)
        self.runtime.provide(self.name, self.build)

    def build(self, b):
        """The prefix stage for the current registrations."""
        layer = ExactLayer(b.device, b.schema)
        layer.enable_prefix(self.options.prefix_min_literal)
        return PrefixStage(layer, b.resolver, b.wake)


class ClassifierStep(SwitchboardStep):
    """Pipeline step: dispatch confident classifier picks."""
    name = "classifier"

    def setup(self):
        self.options = options_from(ClassifierOptions, self.config)
        self.synthetic = data.jsonl(SYNTHETIC_FILE)
        self.runtime.provide(self.name, self.build, slow=True)

    def build(self, b):
        """The classifier stage for the current registrations, trained or loaded from the model cache."""
        synthetic = list(self.synthetic)
        if self.options.dialog_negatives:
            synthetic += harvest_dialog_negatives(self.runtime.dialog_lines(b.lang), b.definitions, b.lang)
        rows = training_data(b.device, b.schema, synthetic)
        classifier, cached = self.runtime.models.get_or_build("classifier", [model_fingerprint(), rows],
                                                              lambda: train_isolated(rows), Classifier)
        LOG.info("switchboard classifier %s (%d training rows)", "loaded from cache" if cached else "trained",
                 len(rows))
        return ClassifierStage(classifier, b.resolver, self.options)


class LLMStep(SwitchboardStep):
    """Pipeline step: route to skill tools with an LLM, behind the addressee gate."""
    name = "llm"
    off_thread_in_shadow = True

    def setup(self):
        if not self.config.get("url"):
            # OVOS instantiates every installed pipeline plugin, listed in the pipeline or not: stay inert.
            LOG.warning("ovos-switchboard-llm-pipeline-plugin has no `url` configured; the LLM step never matches")
            return
        self.llm = LLMClient(options_from(LLMEndpoint, self.config), data.prompt("fallback_strict.txt"),
                             data.prompt("addressee_gate.txt"))
        self.options = options_from(LLMOptions, self.config)
        self.runtime.provide(self.name, self.build)

    def build(self, b):
        """Build the LLM stage's tools and shortlist from the current registrations."""
        tools = build_tools(b.definitions, b.lang)
        return LLMStage(self.llm, Shortlist(tools, b.device.examples), b.schema, b.resolver, self.options)


class DeciderStep(SwitchboardStep):
    """Pipeline step: route to a shortlisted skill intent with a local typed-decision model. An alternative to the
    LLM step where no LLM host is available; place it where the LLM step would go (or before it, with
    on_unsure "pass")."""
    name = "decider"
    EXTRA = ("torch", "transformers")

    def setup(self):
        from importlib.util import find_spec
        if not all(find_spec(m) for m in self.EXTRA):
            # OVOS instantiates every installed pipeline plugin: without the "decider" extra, stay inert.
            LOG.warning("ovos-switchboard-decider-pipeline-plugin needs the 'decider' extra (torch, transformers); "
                        "the decider step never matches")
            return
        self.decider = None
        self.options = options_from(DeciderOptions, self.config)
        self.runtime.provide(self.name, self.build, slow=True)

    def build(self, b):
        """Build the decider stage's shortlist from the current registrations; load the model on first build, off
        the boot path (the first ever build downloads it)."""
        if self.decider is None:
            from ovos_switchboard.adapters.decider_model import DeciderModel, TasksourceDecider
            self.decider = TasksourceDecider(options_from(DeciderModel, self.config))
        tools = build_tools(b.definitions, b.lang)
        return DeciderStage(self.decider, Shortlist(tools, b.device.examples), b.schema, b.resolver, self.options)
