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
from ovos_switchboard.core.options import ClassifierOptions, EchoOptions, LLMOptions
from ovos_switchboard.core.ports import LLMUnavailable
from ovos_switchboard.core.schema import UNMAPPED
from ovos_switchboard.core.tools import META_PREFIX
from ovos_switchboard.core.verdict import Outcome, Verdict, declined, passed


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
    """A DISPATCH verdict for `intent` if the device has a skill label for it, else PASS."""
    label, data, how, slots = resolver.resolve(intent, utterance, loose_slots)
    if how:
        details["label_how"] = how
    if not label:
        return passed(stage, "no skill label", intent=intent, slots=slots, details=details)
    return Verdict(stage, Outcome.DISPATCH, intent, label, data, slots, details=details)


class EchoStage:
    """Declines utterances that repeat the assistant's recent speech."""
    name = "echo"

    def __init__(self, options=EchoOptions()):
        self.options = options

    def decide(self, _utterance, context):
        """DECLINE when the echo run reaches echo_run_min, else PASS."""
        run = context.get("echo_run", 0)
        if run >= self.options.echo_run_min:
            return declined(self.name, "echo", details={"echo_run": run})
        return passed(self.name, "no echo")


class ExactStage:
    """Dispatches utterances that match one intent's declared templates."""
    name = "exact"

    def __init__(self, exact_layer, resolver):
        self.exact, self.resolver = exact_layer, resolver

    def decide(self, utterance, _context=None):
        """DISPATCH to the unique template owner, else PASS."""
        owner = self.exact.route(utterance.lower())
        if not owner:
            return passed(self.name, "no unique template owner")
        return resolved(self.name, owner, utterance, self.resolver, loose_slots=False, confidence=1.0)


class ClassifierStage:
    """Dispatches confident classifier picks; applies the decline-class policy."""
    name = "classifier"

    def __init__(self, classifier, resolver, options=ClassifierOptions()):
        self.classifier, self.resolver, self.options = classifier, resolver, options

    def decide(self, utterance, context):
        """DISPATCH when confident, DECLINE or PASS for the decline class, else PASS."""
        ranked = self.classifier.rank(utterance.lower())
        top, conf = ranked[0]
        details = {"confidence": round(conf, 4), "top3": [c for c, _ in ranked[:3]]}
        if top == DECLINE:
            details["decline_class"] = True
            if self._pass_onward(context):
                return passed(self.name, "decline class, assistant active", details=details)
            return declined(self.name, "decline_class", details=details)
        if top.startswith(UNMAPPED):
            return passed(self.name, "unmapped intent", details=details)
        if conf < self.options.threshold:
            return passed(self.name, "low confidence", details=details)
        return resolved(self.name, top, utterance, self.resolver, **details)

    def _pass_onward(self, context):
        gap, policy = context.get("gap_s"), self.options.route_decline_class
        return policy == "always" or (policy == "active" and gap is not None and gap < self.options.active_s)


class LLMStage:
    """Asks the LLM to pick a skill tool, gated on whether the speech was addressed at all."""
    name = "llm"

    def __init__(self, llm, shortlist, schema, resolver, options=LLMOptions()):
        self.llm, self.shortlist, self.schema = llm, shortlist, schema
        self.resolver, self.options = resolver, options

    def decide(self, utterance, _context=None):
        """DISPATCH a skill tool; DECLINE on ignore, a failed gate or no LLM; PASS meta tools."""
        t0 = time.perf_counter()
        try:
            res = self.llm.route(utterance, self.shortlist.tools(utterance, self.options.shortlist_k))
            gate = self.llm.p_addressed(utterance)
        except LLMUnavailable as e:
            return declined(self.name, "llm_unavailable",
                            details={"error": str(e), "llm_ms": round((time.perf_counter() - t0) * 1000, 1)})
        tool, args = res["tool"], res["args"]
        intent, slots = self._tool_intent(tool, args)
        details = {"llm_tool": tool, "tool_args": args, "p_addressed": round(gate["p_yes"], 4),
                   "llm_ms": res["ms"] + gate["ms"]}
        if gate["p_yes"] < self.options.gate:
            return declined(self.name, "addressee_gate", intent=intent, details=details)
        if tool == "ignore":
            return declined(self.name, "ignore", details=details)
        label = self.shortlist.labels.get(tool, META_PREFIX)
        if label.startswith(META_PREFIX) or not intent:
            return passed(self.name, f"llm chose {tool}", intent=intent, slots=slots, details=details)
        data = args or self.resolver.keyword_data(label, utterance)
        return Verdict(self.name, Outcome.DISPATCH, intent, label, data, slots, details=details)

    def _tool_intent(self, tool, args):
        """(canonical intent or None, canonical slots) for a tool call."""
        strings = {k: v for k, v in args.items() if isinstance(v, str)}
        if tool in META_CANON:
            return META_CANON[tool], strings
        label = self.shortlist.labels[tool]
        canon, renames = self.schema.label_map.get(label, (UNMAPPED + label, {}))
        return canon, {renames.get(k, k): v for k, v in strings.items()}
