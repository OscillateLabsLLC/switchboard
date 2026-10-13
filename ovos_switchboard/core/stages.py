"""The routing stages. Each decides one utterance on its own inputs and never consults another stage: the pipeline's
order guarantees a stage only sees what earlier stages passed on.

  EchoStage        the utterance repeats the assistant's recent speech -> decline
  ExactStage       a declared template phrase with exactly one owner -> dispatch
  ClassifierStage  confident pick -> dispatch; the decline class -> decline or pass onward, per policy
  LLMStage         skill tool + addressee gate -> dispatch; ignore / failed gate / unreachable -> decline;
                   meta tools (ask_knowledge, stop, unsupported) -> pass, so the host's own stages answer

`context` is {"gap_s": seconds since the assistant last spoke or None, "echo_run": int}.
"""
import time

from ovos_switchboard.core.device import DECLINE
from ovos_switchboard.core.ports import LLMUnavailable
from ovos_switchboard.core.schema import UNMAPPED
from ovos_switchboard.core.tools import META_PREFIX
from ovos_switchboard.core.verdict import Outcome, Verdict, declined, passed

DECLINE_POLICIES = ("never", "active", "always")  # when decline-class rows pass on to later stages (the LLM)
META_CANON = {"ask_knowledge": "knowledge.ask", "stop": "stop", "unsupported": None, "ignore": None}


class LabelResolver:
    """Canonical intent -> the device's skill label, the skill-native match_data, and canonical slots."""

    def __init__(self, labels, slots):
        self.labels, self.slots = labels, slots

    def resolve(self, intent, utterance, loose_slots=True):
        """(label or None, match_data, how, slots)"""
        slots = self.slots.extract(intent, utterance.lower(), loose=loose_slots)
        if intent.startswith(UNMAPPED):
            return intent.removeprefix(UNMAPPED), {}, None, slots
        label, data, how = self.labels.select(intent, utterance)
        return label, data, how, slots

    def keyword_data(self, label, utterance):
        """Keyword tools carry no parameters; the vocab the skill reads (message.data["Location"]) comes from
        evaluating its rule on the utterance."""
        rule = self.labels.adapt.get(label)
        hit = rule.evaluate(utterance.lower().strip()) if rule else None
        return hit[1] if hit else {}


def resolved(stage, intent, utterance, resolver, loose_slots=True, **details):
    label, data, how, slots = resolver.resolve(intent, utterance, loose_slots)
    if how:
        details["label_how"] = how
    if not label:
        return passed(stage, "no skill label", intent=intent, slots=slots, details=details)
    return Verdict(stage, Outcome.DISPATCH, intent, label, data, slots, details=details)


class EchoStage:
    name = "echo"

    def __init__(self, echo_run_min=4):
        self.echo_run_min = echo_run_min

    def decide(self, utterance, context):
        run = context.get("echo_run", 0)
        if run >= self.echo_run_min:
            return declined(self.name, "echo", details={"echo_run": run})
        return passed(self.name, "no echo")


class ExactStage:
    name = "exact"

    def __init__(self, exact_layer, resolver):
        self.exact, self.resolver = exact_layer, resolver

    def decide(self, utterance, context=None):
        owner = self.exact.route(utterance.lower())
        if not owner:
            return passed(self.name, "no unique template owner")
        return resolved(self.name, owner, utterance, self.resolver, loose_slots=False, confidence=1.0)


class ClassifierStage:
    name = "classifier"

    def __init__(self, classifier, resolver, threshold=0.5, route_decline_class="active", active_s=120):
        if route_decline_class not in DECLINE_POLICIES:
            raise ValueError(f"route_decline_class must be one of {DECLINE_POLICIES}, got {route_decline_class!r}")
        self.classifier, self.resolver = classifier, resolver
        self.threshold, self.policy, self.active_s = threshold, route_decline_class, active_s

    def decide(self, utterance, context):
        ranked = self.classifier.rank(utterance.lower())
        top, conf = ranked[0]
        details = {"confidence": round(conf, 4), "top3": [c for c, _ in ranked[:3]]}
        if top == DECLINE:
            details["decline_class"] = True
            if self._pass_onward(context):
                return passed(self.name, "decline class, assistant active", details=details)
            return declined(self.name, "decline_class", details=details)
        if top.startswith(UNMAPPED) or conf < self.threshold:
            return passed(self.name, "low confidence", details=details)
        return resolved(self.name, top, utterance, self.resolver, **details)

    def _pass_onward(self, context):
        gap = context.get("gap_s")
        return self.policy == "always" or (self.policy == "active" and gap is not None and gap < self.active_s)


class LLMStage:
    name = "llm"

    def __init__(self, llm, shortlist, tools_doc, schema, resolver, gate=0.9, shortlist_k=12):
        self.llm, self.shortlist, self.tools_doc, self.schema = llm, shortlist, tools_doc, schema
        self.resolver, self.gate, self.k = resolver, gate, shortlist_k

    def decide(self, utterance, context=None):
        t0 = time.perf_counter()
        try:
            res = self.llm.route(utterance, self.shortlist.tools(utterance, self.k))
            gate = self.llm.p_addressed(utterance)
        except LLMUnavailable as e:
            return declined(self.name, "llm_unavailable",
                            details={"error": str(e), "llm_ms": round((time.perf_counter() - t0) * 1000, 1)})
        tool, args = res["tool"], res["args"]
        intent, slots = self._tool_intent(tool, args)
        details = {"llm_tool": tool, "tool_args": args, "p_addressed": round(gate["p_yes"], 4),
                   "llm_ms": res["ms"] + gate["ms"]}
        if gate["p_yes"] < self.gate:
            return declined(self.name, "addressee_gate", intent=intent, details=details)
        if tool == "ignore":
            return declined(self.name, "ignore", details=details)
        label = self.tools_doc["labels"].get(tool, META_PREFIX)
        if label.startswith(META_PREFIX) or not intent:
            return passed(self.name, f"llm chose {tool}", intent=intent, slots=slots, details=details)
        data = args or self.resolver.keyword_data(label, utterance)
        return Verdict(self.name, Outcome.DISPATCH, intent, label, data, slots, details=details)

    def _tool_intent(self, tool, args):
        """(canonical intent or None, canonical slots) for a tool call."""
        strings = {k: v for k, v in args.items() if isinstance(v, str)}
        if tool in META_CANON:
            return META_CANON[tool], strings
        label = self.tools_doc["labels"][tool]
        canon, renames = self.schema.label_map.get(label, (UNMAPPED + label, {}))
        return canon, {renames.get(k, k): v for k, v in strings.items()}
