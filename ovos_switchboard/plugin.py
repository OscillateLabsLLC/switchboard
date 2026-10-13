"""Switchboard as an OVOS pipeline plugin (OVOS-PIPELINE-1 ConfidenceMatcherPipeline).

Tiers (recommended pipeline order in README.md):
  high    echo guard -> decline; exact template phrase -> dispatch
  medium  classifier accept -> dispatch; decline class (not sent to the LLM by policy) -> decline
  low     LLM: skill tool -> dispatch; ignore / failed addressee gate / LLM unreachable -> decline;
          ask_knowledge, stop, unsupported -> no match, so OVOS's own common_qa / fallback stages answer
A decline dispatches to the Hangup skill, which speaks nothing and records why.

mode "shadow" (default) routes every utterance at the high tier, logs the decision and its latency, and returns no
match, so the device behaves exactly as before. mode "active" acts on the decisions.
"""
import threading
import time
from dataclasses import dataclass, field

from ovos_plugin_manager.templates.pipeline import ConfidenceMatcherPipeline, IntentHandlerMatch

from ovos_switchboard.context import SpeechContext
from ovos_switchboard.decisions import DecisionCache, DecisionLog
from ovos_switchboard.llm import LLMClient, LLMUnavailable, bundled_prompt
from ovos_switchboard.registry import IntentRegistry, session_of
from ovos_switchboard.router import RouterConfig
from ovos_switchboard.schema import Schema
from ovos_switchboard.trainer import RouterTrainer

try:
    from ovos_utils.log import LOG
except ImportError:  # pragma: no cover
    import logging
    LOG = logging.getLogger("switchboard")

HANGUP_SKILL_ID = "ovos-skill-hangup.oscillatelabsllc"
HANGUP_INTENT = "decline"  # interface contract with Hangup (HangupSkill.DECLINE_INTENT)
MODES = ("shadow", "active")
SPEAK_TOPICS = ("speak", "ovos.utterance.speak")
OUTCOME_TOPICS = ("ovos.intent.matched", "ovos.intent.unmatched", "complete_intent_failure")
ROUTER_FIELDS = ("intent", "stage", "label", "match_data", "confidence", "top3", "llm_tool", "tool_args",
                 "p_addressed", "llm_ms", "label_how", "decline_class")


@dataclass
class Settings:
    mode: str = "shadow"
    lang: str = "en-us"
    hangup_skill_id: str = HANGUP_SKILL_ID
    high_risk: str = "defer"            # "defer": never act on high-risk intents (no confirm flow yet); "allow"
    retrain_debounce_s: float = 5.0
    dialog_negatives: bool = True
    decision_log: bool = True
    decision_log_path: str = ""
    decision_log_max_bytes: int = 5_000_000
    schema_path: str = ""
    llm: dict = field(default_factory=dict)   # url, model, api_key, timeout_s; empty = no LLM stage
    router: RouterConfig = field(default_factory=RouterConfig)

    @classmethod
    def from_config(cls, config):
        config = dict(config or {})
        router_keys = RouterConfig.__dataclass_fields__.keys()
        router = RouterConfig(**{k: config.pop(k) for k in list(config) if k in router_keys})
        known = {k: v for k, v in config.items() if k in cls.__dataclass_fields__}
        settings = cls(router=router, **known)
        if settings.mode not in MODES:
            raise ValueError(f"switchboard mode must be one of {MODES}, got {settings.mode!r}")
        return settings


class SwitchboardPipeline(ConfidenceMatcherPipeline):
    def __init__(self, bus=None, config=None):
        super().__init__(bus, config)
        self.settings = s = Settings.from_config(self.config)
        self.speech = SpeechContext()
        self.cache = DecisionCache()
        self.decision_log = DecisionLog(s.decision_log_path or None, s.decision_log_max_bytes, s.decision_log)
        self.registry = IntentRegistry(self.bus, on_change=lambda: self.trainer.schedule())
        self.trainer = RouterTrainer(self.registry, Schema.load(s.schema_path or None), s.router, self._make_llm(),
                                     s.lang, s.retrain_debounce_s, s.dialog_negatives)
        self.registry.add_ocp_templates(s.lang)
        for topic in SPEAK_TOPICS:
            self.bus.on(topic, self._on_speak)
        for topic in OUTCOME_TOPICS:
            self.bus.on(topic, self._on_outcome)
        threading.Thread(target=self._startup, daemon=True, name="switchboard-startup").start()
        LOG.info(f"Switchboard loaded in {s.mode} mode, LLM {'on' if self.trainer.llm else 'off'}")

    def _make_llm(self):
        llm = self.settings.llm
        if not llm.get("url"):
            return None
        return LLMClient(llm["url"], llm.get("model", ""), bundled_prompt("fallback_strict.txt"),
                         bundled_prompt("addressee_gate.txt"), float(llm.get("timeout_s", 10)), llm.get("api_key"))

    def _startup(self):
        try:
            self.registry.sync()
        except Exception:
            LOG.exception("initial intent sync failed; relying on registration broadcasts")
        self._check_hangup()

    def _check_hangup(self):
        from ovos_bus_client.message import Message
        reply = self.bus.wait_for_response(Message("ovos.skills.list", {}), "ovos.skills.list.response", timeout=10)
        if reply and reply.data.get("ok"):
            loaded = {sk["skill_id"] for sk in reply.data.get("skills") or []}
            if self.settings.hangup_skill_id not in loaded:
                LOG.error(f"{self.settings.hangup_skill_id} is not loaded: Switchboard's declines will go nowhere. "
                          "Install the Hangup skill.")

    def shutdown(self):
        self.trainer.shutdown()
        self.registry.shutdown()
        for topic in SPEAK_TOPICS:
            self.bus.remove(topic, self._on_speak)
        for topic in OUTCOME_TOPICS:
            self.bus.remove(topic, self._on_outcome)

    # ---- bus observers ----

    def _on_speak(self, message):
        self.speech.heard_assistant(message.data.get("utterance", ""), session_of(message))

    def _on_outcome(self, message):
        data = message.data
        self.decision_log.write({"event": message.msg_type, "utterance": data.get("utterance"),
                                 "skill_id": data.get("skill_id"), "intent_name": data.get("intent_name"),
                                 "pipeline_id": data.get("pipeline_id"), "session_id": session_of(message)})

    # ---- tiers ----

    def match_high(self, utterances, lang, message):
        entry = self._fast(utterances, lang, message)
        if not entry:
            return None
        if self.settings.mode == "shadow":
            self._shadow(entry)
            return None
        d = entry["decision"]
        if d["stage"] == "echo":
            return self._decline(entry, "echo")
        return self._dispatch(entry) if d["stage"] == "exact" else None

    def match_medium(self, utterances, lang, message):
        entry = self._fast(utterances, lang, message)
        if not entry:
            return None
        if self.settings.mode == "shadow":
            self._shadow(entry)
            return None
        d = entry["decision"]
        if d["stage"] in ("classifier", "prefix") and d.get("intent"):
            return self._dispatch(entry)
        if d.get("decline_class") and not d.get("needs_llm"):
            return self._decline(entry, "decline_class")
        return None

    def match_low(self, utterances, lang, message):
        entry = self._fast(utterances, lang, message)
        if not entry:
            return None
        if self.settings.mode == "shadow":
            self._shadow(entry)
            return None
        if not entry["decision"].get("needs_llm"):
            return None
        try:
            self._run_llm(entry)
        except LLMUnavailable as e:
            LOG.warning(f"LLM unavailable, declining: {e}")
            return self._decline(entry, "llm_unavailable")
        d = entry["decision"]
        if d.get("label"):
            return self._dispatch(entry)
        if d["stage"] == "llm_gate" or d.get("llm_tool") == "ignore":
            return self._decline(entry, "ignore" if d.get("llm_tool") == "ignore" else "addressee_gate")
        self._record(entry, "defer", d.get("llm_tool"))
        return None

    # ---- routing ----

    def _fast(self, utterances, lang, message):
        router = self.trainer.router
        if not router or not utterances or not self._serves(lang):
            return None
        utterance, session_id = utterances[0], session_of(message)
        key = (session_id, utterance)
        entry = self.cache.get(key)
        if entry is None:
            t0 = time.perf_counter()
            context = self.speech.features(utterance, session_id)
            decision = router.route_fast(utterance, context)
            entry = {"utterance": utterance, "session_id": session_id, "lang": lang, "context": context,
                     "decision": decision, "router": router, "fast_ms": round((time.perf_counter() - t0) * 1000, 1)}
            self.cache.put(key, entry)
        return entry

    def _serves(self, lang):
        return (lang or "").lower().split("-")[0] == self.settings.lang.lower().split("-")[0]

    def _run_llm(self, entry):
        if "llm_done" not in entry:
            entry["decision"] = entry["router"].route_llm(entry["utterance"], entry["decision"])
            entry["llm_done"] = True

    def _shadow(self, entry):
        if entry.get("shadowed"):
            return
        entry["shadowed"] = True
        if not entry["decision"].get("needs_llm"):
            self._record(entry, "shadow")
            return

        def finish():
            try:
                self._run_llm(entry)
                self._record(entry, "shadow")
            except LLMUnavailable as e:
                self._record(entry, "shadow", f"llm_unavailable: {e}")
        threading.Thread(target=finish, daemon=True, name="switchboard-shadow-llm").start()

    # ---- outcomes ----

    def _dispatch(self, entry):
        d = entry["decision"]
        label = d.get("label")
        if not label or ":" not in label:
            self._record(entry, "defer", "no skill label")
            return None
        if self.settings.high_risk == "defer" and self.trainer.schema.risk(d.get("intent")) == "high":
            self._record(entry, "defer", "high-risk intent, no confirm flow")
            return None
        self._record(entry, "dispatch")
        return IntentHandlerMatch(match_type=label, match_data=dict(d.get("match_data") or {}),
                                  skill_id=label.split(":", 1)[0], utterance=entry["utterance"])

    def _decline(self, entry, reason):
        d = entry["decision"]
        details = {k: d[k] for k in ("confidence", "p_addressed", "llm_tool") if d.get(k) is not None}
        self._record(entry, "decline", reason)
        skill_id = self.settings.hangup_skill_id
        return IntentHandlerMatch(match_type=f"{skill_id}:{HANGUP_INTENT}", skill_id=skill_id,
                                  match_data={"reason": reason, "stage": d.get("stage"), **details},
                                  utterance=entry["utterance"])

    def _record(self, entry, action, reason=None):
        d = entry["decision"]
        self.decision_log.write({
            "event": "switchboard.decision", "mode": self.settings.mode, "action": action, "reason": reason,
            "utterance": entry["utterance"], "session_id": entry["session_id"], "lang": entry["lang"],
            **entry["context"], "fast_ms": entry["fast_ms"], **{k: d[k] for k in ROUTER_FIELDS if k in d}})
