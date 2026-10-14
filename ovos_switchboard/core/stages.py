"""The routing stages. Each decides one utterance on its own inputs and never consults another stage: the pipeline's
order guarantees a stage only sees what earlier stages passed on. One exception, by measurement: ExactStage may read
the classifier MODEL's ranking (a port, not the classifier stage's verdict) to overrule catch-all templates.

  EchoStage        the utterance repeats the assistant's recent speech -> decline
  ExactStage       a declared template phrase with exactly one owner -> dispatch
  PrefixStage      a slot-free template phrase followed by more speech, one owner -> dispatch
  ClassifierStage  confident pick -> dispatch; the decline class -> decline or pass onward, per policy
  LLMStage         skill tool + addressee gate -> dispatch; ignore / failed gate / unreachable -> decline;
                   meta tools (ask_knowledge, stop, unsupported) -> pass, so the host's own stages answer
  DeciderStage     a typed-decision model picks a shortlisted skill intent -> dispatch (no tool arguments: the
                   skill gets Switchboard's own match data); "none of these" or unsure -> decline, or pass if
                   configured; meta choices (ask_knowledge, stop) -> pass, as for the LLM

`context` is {"gap_s": seconds since the assistant last spoke or None, "echo_run": int}.
"""
import time

from ovos_switchboard.core.device import DECLINE
from ovos_switchboard.core.options import ClassifierOptions, DeciderOptions, EchoOptions, ExactOptions, LLMOptions
from ovos_switchboard.core.ports import LLMUnavailable
from ovos_switchboard.core.schema import UNMAPPED
from ovos_switchboard.core.tools import DECIDER_NONE, DECIDER_QUESTION, META_PREFIX, decider_options
from ovos_switchboard.core.verdict import Outcome, Verdict, declined, passed
from ovos_switchboard.core.wake import strip_wake_phrase


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
    """Dispatches utterances that match one intent's declared templates.

    `ranker(utterance)` -> [(intent, confidence), ...] best first, or None while no classifier is built. Only
    catch-all matches ("play {query}") consult it: a confident pick of another intent overrules them ("play
    something like glados" is the easter egg; Neon's own "call me Neon, I am ..." is not a call)."""
    name = "exact"

    def __init__(self, exact_layer, resolver, wake=None, ranker=None,  # pylint: disable=R0913
                 options=ExactOptions()):
        self.exact, self.resolver, self.wake = exact_layer, resolver, wake
        self.ranker, self.options = ranker, options

    def decide(self, utterance, _context=None):
        """DISPATCH to the unique template owner, else PASS. A catch-all owner yields to a confident classifier."""
        utterance = strip_wake_phrase(utterance, self.wake)
        owner, catch_all = self.exact.match(utterance.lower())
        if not owner:
            return passed(self.name, "no unique template owner")
        overruled = self._overruled(owner, utterance) if catch_all else None
        if overruled:
            return passed(self.name, "catch-all template overruled by the classifier", details=overruled)
        return resolved(self.name, owner, utterance, self.resolver, loose_slots=False, confidence=1.0)

    def _overruled(self, owner, utterance):
        """Details when the classifier ranks another intent first with catch_all_yield confidence, else None."""
        threshold = self.options.catch_all_yield
        ranking = self.ranker(utterance.lower()) if self.ranker and threshold > 0 else None
        if not ranking:
            return None
        top, conf = ranking[0]
        if top == owner or conf < threshold:
            return None
        return {"catch_all": owner, "overruled_by": top, "confidence": round(conf, 4)}


class PrefixStage:
    """Dispatches trailing speech after a declared phrase ("make me laugh hey dad"): slot-free templates of
    enough literal words, matched as a prefix. Measured in the harness but not yet validated on live traffic."""
    name = "prefix"

    def __init__(self, prefix_layer, resolver, wake=None):
        self.prefix, self.resolver, self.wake = prefix_layer, resolver, wake

    def decide(self, utterance, _context=None):
        """DISPATCH to the unique owner of a matching template prefix, else PASS."""
        utterance = strip_wake_phrase(utterance, self.wake)
        owner = self.prefix.prefix_route(utterance.lower())
        if not owner:
            return passed(self.name, "no unique prefix owner")
        return resolved(self.name, owner, utterance, self.resolver)


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


class DeciderStage:
    """A typed-decision model picks one shortlisted intent, in plain words, with a probability: a fast alternative
    to the LLM stage for hosts without one. It never sees tool arguments, so dispatch carries Switchboard's own
    match data."""
    name = "decider"

    def __init__(self, decider, shortlist, schema, resolver, options=DeciderOptions()):
        self.decider, self.shortlist, self.schema = decider, shortlist, schema
        self.resolver, self.options = resolver, options

    def decide(self, utterance, _context=None):
        """DISPATCH a confident skill pick; PASS meta picks; "none of these" or below threshold -> on_unsure."""
        options = decider_options(self.shortlist.tools(utterance, self.options.shortlist_k))
        res = self.decider.choose(utterance, DECIDER_QUESTION, options)
        choice = res["choice"]
        p = res["probabilities"][choice]
        details = {"decider_choice": choice, "decider_p": round(p, 4), "decider_ms": res["ms"]}
        if choice == DECIDER_NONE or p < self.options.threshold:
            reason = "none of these" if choice == DECIDER_NONE else "below threshold"
            unsure = passed if self.options.on_unsure == "pass" else declined
            return unsure(self.name, f"decider: {reason}", details=details)
        if choice in META_CANON:
            return passed(self.name, f"decider chose {choice}", intent=META_CANON[choice], details=details)
        label = self.shortlist.labels[choice]
        intent, _ = self.schema.label_map.get(label, (UNMAPPED + label, {}))
        return Verdict(self.name, Outcome.DISPATCH, intent, label, self.resolver.keyword_data(label, utterance), {},
                       details=details)
