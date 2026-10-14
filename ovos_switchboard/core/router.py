"""The stages composed in the measured order, as one call: how the intent-eval harness scored Switchboard.

The OVOS plugin runs each stage as its own pipeline step instead; this composition exists so the regression gate
(scripts/check_harness.py) verifies the very stage objects the steps run. Output is the harness decision shape.

  echo -> exact -> classifier -> [prefix, off by default; unvalidated] -> LLM
"""
from dataclasses import dataclass, field

from ovos_switchboard.core.classifier import Classifier
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.labels import LabelSelector
from ovos_switchboard.core.options import ClassifierOptions, EchoOptions, LLMOptions
from ovos_switchboard.core.shortlist import Shortlist
from ovos_switchboard.core.slots import SlotExtractor
from ovos_switchboard.core.stages import ClassifierStage, EchoStage, ExactStage, LabelResolver, LLMStage, PrefixStage
from ovos_switchboard.core.verdict import Outcome
from ovos_switchboard.core.wake import wake_phrase_pattern


DETAIL_KEYS = ("confidence", "top3", "label_how", "decline_class")


@dataclass(frozen=True)
class RouterConfig:
    """Options for every stage. prefix_min_literal 0 turns the prefix stage off; no wake phrases strips none."""
    echo: EchoOptions = field(default_factory=EchoOptions)
    classifier: ClassifierOptions = field(default_factory=ClassifierOptions)
    llm: LLMOptions = field(default_factory=LLMOptions)
    prefix_min_literal: int = 0
    wake_phrases: tuple = ()
    device_classes_only: bool = True


class Router:
    """All stages in measured order, returning the harness decision shape."""
    def __init__(self, device, schema, tools_doc, *, synthetic=(), llm=None, config=None):  # pylint: disable=R0913
        c = self.cfg = config or RouterConfig()
        self.resolver = LabelResolver(LabelSelector(device, schema), SlotExtractor(device, schema))
        exact_layer = ExactLayer(device, schema)
        wake = wake_phrase_pattern(c.wake_phrases)
        self.prefix = None
        if c.prefix_min_literal:
            exact_layer.enable_prefix(c.prefix_min_literal)
            self.prefix = PrefixStage(exact_layer, self.resolver, wake)
        self.echo = EchoStage(c.echo)
        self.exact = ExactStage(exact_layer, self.resolver, wake)
        classifier = Classifier.for_device(device, schema, synthetic, c.device_classes_only)
        self.classifier = ClassifierStage(classifier, self.resolver, c.classifier)
        self.llm = LLMStage(llm, Shortlist(tools_doc, device.examples), schema, self.resolver, c.llm) if llm else None

    def route(self, utterance, context=None):
        """The harness decision for one utterance: echo, exact, classifier, then the LLM."""
        context = context or {}
        if self.echo.decide(utterance, context).decisive:
            return {"intent": None, "stage": "echo"}
        v = self.exact.decide(utterance, context)
        if v.intent:
            return self._as_decision(v)
        v = self.classifier.decide(utterance, context)
        if v.intent:
            return self._as_decision(v)
        base = {"intent": None, "stage": "classifier", **{k: v.details[k] for k in DETAIL_KEYS if k in v.details}}
        if self.prefix:
            v_prefix = self.prefix.decide(utterance, context)
            if v_prefix.intent:
                return self._as_decision(v_prefix, base)
        if self.llm and v.outcome is Outcome.PASS:
            return self._llm_decision(self.llm.decide(utterance, context), base)
        return base

    @staticmethod
    def _as_decision(v, base=None):
        d = {**(base or {}), "intent": v.intent, "stage": v.stage, "slots": v.slots,
             **{k: v.details[k] for k in DETAIL_KEYS if k in v.details}}
        if v.label or v.outcome is Outcome.PASS:
            d.update(label=v.label, match_data=v.match_data)
        return d

    @staticmethod
    def _llm_decision(v, base):
        if v.reason == "llm_unavailable":
            raise RuntimeError(f"LLM unavailable: {v.details.get('error')}")
        gated = v.reason == "addressee_gate"
        d = {**base, "intent": None if gated else v.intent, "slots": {} if gated else v.slots,
             "stage": "llm_gate" if gated else "llm",
             **{k: v.details[k] for k in ("llm_tool", "tool_args", "p_addressed", "llm_ms")}}
        if v.label:
            d.update(label=v.label, match_data=v.match_data)
        return d
