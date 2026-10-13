"""The stages composed in the measured order, as one call: how the intent-eval harness scored Switchboard.

The OVOS plugin runs each stage as its own pipeline step instead; this composition exists so the regression gate
(scripts/check_harness.py) verifies the very stage objects the steps run. Output is the harness decision shape.

  echo -> exact -> classifier -> [prefix, off by default; unvalidated] -> LLM
"""
from dataclasses import dataclass

from ovos_switchboard.core.classifier import Classifier
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.labels import LabelSelector
from ovos_switchboard.core.shortlist import Shortlist
from ovos_switchboard.core.slots import SlotExtractor
from ovos_switchboard.core.stages import ClassifierStage, EchoStage, ExactStage, LabelResolver, LLMStage, resolved
from ovos_switchboard.core.verdict import Outcome

DETAIL_KEYS = ("confidence", "top3", "label_how", "decline_class")


@dataclass
class RouterConfig:
    threshold: float = 0.5
    prefix_min_literal: int = 0
    route_decline_class: str = "active"
    gate: float = 0.9
    shortlist_k: int = 12
    device_classes_only: bool = True
    echo_run_min: int = 4
    active_s: float = 120


class Router:
    def __init__(self, device, schema, tools_doc, synthetic=(), llm=None, config=None):
        c = self.cfg = config or RouterConfig()
        self.resolver = LabelResolver(LabelSelector(device, schema), SlotExtractor(device, schema))
        self.exact_layer = ExactLayer(device, schema)
        if c.prefix_min_literal:
            self.exact_layer.enable_prefix(c.prefix_min_literal)
        self.echo = EchoStage(c.echo_run_min)
        self.exact = ExactStage(self.exact_layer, self.resolver)
        self.classifier = ClassifierStage(Classifier(device, schema, synthetic, c.device_classes_only), self.resolver,
                                          c.threshold, c.route_decline_class, c.active_s)
        self.llm = LLMStage(llm, Shortlist(tools_doc, device.examples), tools_doc, schema, self.resolver,
                            c.gate, c.shortlist_k) if llm else None

    def route(self, utterance, context=None):
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
        if self.cfg.prefix_min_literal:
            owner = self.exact_layer.prefix_route(utterance.lower())
            if owner:
                return self._as_decision(resolved("prefix", owner, utterance, self.resolver), base)
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
