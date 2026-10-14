"""Composition root shared by the Switchboard pipeline steps on one bus.

OVOS instantiates each step separately; they share one runtime per bus (reference counted):
  - the live intent registry (bus adapter) and a debounced background rebuild of whatever the steps asked for:
    each step provides a builder, so a step that isn't installed costs nothing (no classifier training, no shortlist)
  - the assistant's speech, for echo and "assistant active" context
  - the decision log, and OVOS's actual outcomes for comparison
  - a per-utterance record: high-risk holds, and in shadow mode which step already settled the utterance
Shared settings live under `intents.switchboard` in mycroft.conf.
"""
import threading
import time
from dataclasses import dataclass, field, replace
from typing import ClassVar

from ovos_switchboard.adapters.bus_registry import IntentRegistry, session_of
from ovos_switchboard.adapters.decision_log import DecisionLog
from ovos_switchboard.adapters.dialog_files import installed_dialog_lines
from ovos_switchboard.adapters.model_cache import ModelCache
from ovos_switchboard.core.context import SpeechContext
from ovos_switchboard.core.device import Device
from ovos_switchboard.core.labels import LabelSelector
from ovos_switchboard.core.options import SpeechOptions, options_from
from ovos_switchboard.core.schema import Schema
from ovos_switchboard.core.slots import SlotExtractor
from ovos_switchboard.core.stages import LabelResolver
from ovos_switchboard.core.wake import wake_phrase_pattern


try:
    from ovos_utils.log import LOG
except ImportError:  # pragma: no cover
    import logging
    LOG = logging.getLogger("switchboard")

MODES = ("shadow", "active")
SPEAK_TOPICS = ("speak", "ovos.utterance.speak")
OUTCOME_TOPICS = ("ovos.intent.matched", "ovos.intent.unmatched", "complete_intent_failure")
SKILLS_INITIALIZED = "mycroft.skills.initialized"
SKILLS_IS_READY = "mycroft.skills.is_ready"
SWITCHBOARD_IS_READY = "mycroft.switchboard.is_ready"  # add "switchboard" to boot-finished's ready_settings


@dataclass(frozen=True)
class Settings:  # pylint: disable=too-many-instance-attributes  # one field per config key
    """The shared `intents.switchboard` section."""
    mode: str = "shadow"
    lang: str = "en-us"
    hangup_skill_id: str = "ovos-skill-hangup.oscillatelabsllc"
    high_risk: str = "defer"            # "defer": never act on high-risk intents (no confirm flow yet); "allow"
    retrain_debounce_s: float = 5.0
    bus_timeout_s: float = 10.0         # each startup query to ovos-core (intent sync, skills list)
    record_ttl_s: float = 30.0          # must outlast one full pipeline round
    records_max: int = 64
    decision_log: bool = True
    decision_log_path: str = ""
    decision_log_max_bytes: int = 5_000_000
    model_cache: bool = True            # reuse trained models across restarts while their inputs are unchanged
    model_cache_path: str = ""
    model_cache_max: int = 2            # newest models kept per kind
    schema_path: str = ""
    wake_phrases: tuple = ()            # stripped from the start of utterances; default: the listener's hotwords
    # Skills whose own pipeline step must dispatch them: a Switchboard verdict for one passes to that step instead.
    # OCP's handlers need match data only OCP's matcher adds ("media_type"), so ocp:play crashed when we dispatched.
    delegate_skills: tuple = ("ocp",)
    speech: SpeechOptions = field(default_factory=SpeechOptions)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"switchboard mode must be one of {MODES}, got {self.mode!r}")

    @classmethod
    def from_config(cls, config):
        """From the flat `intents.switchboard` section; the speech options share it."""
        return replace(options_from(cls, config), speech=options_from(SpeechOptions, config))


def listener_wake_phrases():
    """The wake words the listener is listening for ("hey_mycroft"), from mycroft.conf hotwords."""
    try:
        from ovos_config import Configuration
    except ImportError:  # pragma: no cover
        return ()
    hotwords = Configuration().get("hotwords") or {}
    return tuple(name for name, cfg in hotwords.items() if isinstance(cfg, dict) and cfg.get("listen"))


@dataclass(frozen=True)
class Build:
    """What every stage builder may draw on, from one registry snapshot."""
    definitions: dict
    device: Device
    resolver: LabelResolver
    schema: Schema
    lang: str
    wake: object  # compiled leading-wake-phrase pattern, or None


@dataclass
class UtteranceRecord:
    """What the steps have settled about one utterance round."""
    held: bool = False     # a step held a high-risk intent: later steps must not act on this utterance
    delegated: bool = False  # a step handed it to its skill's own pipeline step: later steps must not act either
    decline_class_passed: bool = False  # the classifier said "not for me" but passed it on (assistant active)
    settled_by: str = ""   # shadow mode: the first step with a decisive verdict


class SwitchboardRuntime:  # pylint: disable=too-many-instance-attributes  # the composition root
    """What the Switchboard steps on one bus share: registry, stages, speech, log, records."""
    _instances: ClassVar[dict] = {}
    _class_lock = threading.Lock()

    @classmethod
    def acquire(cls, bus, config=None):
        """The bus's runtime, created on first use; pair with release()."""
        with cls._class_lock:
            rt = cls._instances.get(id(bus))
            if rt is None:
                settings = Settings.from_config(config)
                if not settings.wake_phrases:
                    settings = replace(settings, wake_phrases=listener_wake_phrases())
                rt = cls._instances[id(bus)] = cls(bus, settings)
            rt.refs += 1
            return rt

    def release(self):
        """Drop one reference; the last one shuts the runtime down."""
        with self._class_lock:
            self.refs -= 1
            if self.refs <= 0:
                self._instances.pop(id(self.bus), None)
                self.shutdown()

    def __init__(self, bus, settings, dialog_lines=installed_dialog_lines, start=True):
        self.bus, self.settings, self.refs = bus, settings, 0
        self.schema = Schema.load(settings.schema_path or None)
        self.dialog_lines = dialog_lines
        self.speech = SpeechContext(settings.speech)
        self.wake = wake_phrase_pattern(settings.wake_phrases)
        self.log = DecisionLog(settings.decision_log_path or None, settings.decision_log_max_bytes,
                               settings.decision_log)
        self.models = ModelCache(settings.model_cache_path or None, settings.model_cache_max, settings.model_cache)
        self.builders, self.stages = {}, {}
        self.records, self.records_lock = {}, threading.Lock()
        self.timer, self.timer_lock = None, threading.Lock()
        self.rebuild_lock = threading.Lock()
        self.skills_ready = threading.Event()  # rebuilding mid-load trains on a partial set, so wait for core
        self.registry = IntentRegistry(bus, on_change=self.schedule)
        self.registry.add_ocp_templates(settings.lang)
        for topic in SPEAK_TOPICS:
            bus.on(topic, self._on_speak)
        for topic in OUTCOME_TOPICS:
            bus.on(topic, self._on_outcome)
        bus.on(SKILLS_INITIALIZED, self._on_skills_initialized)
        bus.on(SWITCHBOARD_IS_READY, self._on_is_ready)
        if start:
            threading.Thread(target=self._startup, daemon=True, name="switchboard-startup").start()

    def shutdown(self):
        """Cancel pending rebuilds and detach from the bus."""
        with self.timer_lock:
            if self.timer:
                self.timer.cancel()
        self.registry.shutdown()
        for topic in SPEAK_TOPICS:
            self.bus.remove(topic, self._on_speak)
        for topic in OUTCOME_TOPICS:
            self.bus.remove(topic, self._on_outcome)
        self.bus.remove(SKILLS_INITIALIZED, self._on_skills_initialized)
        self.bus.remove(SWITCHBOARD_IS_READY, self._on_is_ready)

    # ---- stages ----

    def provide(self, name, builder, slow=False):
        """builder(build: Build) -> stage, rebuilt on every registry change. Slow builders (trained models) run
        after the others, so cheap stages are usable while a model trains."""
        self.builders[name] = (builder, slow)
        self.schedule()

    def stage(self, name):
        """The built stage `name`, or None until its first build."""
        return self.stages.get(name)

    def schedule(self, delay=None):
        """Rebuild once registrations have been quiet for `delay` (default retrain_debounce_s), and skills have
        finished loading."""
        if not self.skills_ready.is_set():
            return
        with self.timer_lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(self.settings.retrain_debounce_s if delay is None else delay, self.rebuild)
            self.timer.daemon = True
            self.timer.start()

    def rebuild(self):
        """Build every provided stage from the current registrations, one rebuild at a time."""
        with self.rebuild_lock:  # the snapshot is taken inside the lock, so the newest registrations win
            self._rebuild()

    def _rebuild(self):
        definitions = self.registry.definitions()
        if not definitions or not self.builders:
            return
        t0 = time.perf_counter()
        device = Device.from_definitions(definitions, self.settings.lang)
        resolver = LabelResolver(LabelSelector(device, self.schema), SlotExtractor(device, self.schema))
        build = Build(definitions, device, resolver, self.schema, self.settings.lang, self.wake)
        for name, (builder, _) in sorted(self.builders.items(), key=lambda item: item[1][1]):
            t_stage = time.perf_counter()
            try:
                self.stages[name] = builder(build)
            except Exception:
                LOG.exception("rebuilding the %s stage failed; keeping the previous one", name)
                continue
            LOG.info("switchboard %s stage ready in %.1fs", name, time.perf_counter() - t_stage)
        LOG.info("switchboard rebuilt %s from %d intents in %.1fs", sorted(self.builders),
                 sum(map(len, definitions.values())), time.perf_counter() - t0)

    # ---- context and records ----

    def serves(self, lang):
        """Whether `lang` is the language the stages serve."""
        return (lang or "").lower().split("-")[0] == self.settings.lang.lower().split("-")[0]

    def context(self, utterance, session_id):
        """Addressee context for `utterance` from the assistant's recent speech."""
        return self.speech.features(utterance, session_id)

    def record(self, message, utterance):
        """One record per utterance round. Every matcher in a round receives the same Message object."""
        key, now = (id(message), utterance), time.monotonic()
        with self.records_lock:
            for k in [k for k, (t, _) in self.records.items() if now - t > self.settings.record_ttl_s]:
                del self.records[k]
            if key not in self.records:
                while len(self.records) >= self.settings.records_max:
                    self.records.pop(next(iter(self.records)))
                self.records[key] = (now, UtteranceRecord())
            return self.records[key][1]

    # ---- bus observers ----

    def _startup(self):
        try:
            self.registry.sync(self.settings.bus_timeout_s)
        except Exception:
            LOG.exception("initial intent sync failed; relying on registration broadcasts")
        from ovos_bus_client.message import Message
        reply = self.bus.wait_for_response(Message(SKILLS_IS_READY), f"{SKILLS_IS_READY}.response",
                                           timeout=self.settings.bus_timeout_s)
        if reply and reply.data.get("status"):  # loaded after core finished (a pipeline reload)
            self._skills_loaded()

    def _on_skills_initialized(self, _message):
        threading.Thread(target=self._skills_loaded, daemon=True, name="switchboard-skills-loaded").start()

    def _skills_loaded(self):
        if self.skills_ready.is_set():
            return
        self.skills_ready.set()
        # Everything has registered by now, so the debounce would only make "I am ready" wait longer.
        self.schedule(delay=0)
        self._check_hangup()

    def ready(self):
        """Whether every provided stage has been built at least once."""
        return self.skills_ready.is_set() and bool(self.builders) and set(self.builders) <= set(self.stages)

    def _on_is_ready(self, message):
        # OVOS readiness protocol: ovos-skill-boot-finished asks each name in its ready_settings.
        self.bus.emit(message.response({"status": self.ready()}))

    def _check_hangup(self):
        from ovos_bus_client.message import Message
        reply = self.bus.wait_for_response(Message("ovos.skills.list", {}), "ovos.skills.list.response",
                                           timeout=self.settings.bus_timeout_s)
        if reply and reply.data.get("ok"):
            loaded = {sk["skill_id"] for sk in reply.data.get("skills") or []}
            if self.settings.hangup_skill_id not in loaded:
                LOG.error("%s is not loaded: Switchboard's declines will go nowhere. Install the Hangup skill.",
                          self.settings.hangup_skill_id)

    def _on_speak(self, message):
        self.speech.heard_assistant(message.data.get("utterance", ""), session_of(message))

    def _on_outcome(self, message):
        data = message.data
        self.log.write({"event": message.msg_type, "utterance": data.get("utterance"),
                        "skill_id": data.get("skill_id"), "intent_name": data.get("intent_name"),
                        "pipeline_id": data.get("pipeline_id"), "session_id": session_of(message)})
