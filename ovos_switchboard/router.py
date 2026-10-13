"""The router: route(utterance, context) -> decision, stages in waterfall order.

Early stages are tuned for precision (a wrong accept never reaches a fallback); later stages recover recall.
  1. echo guard     context["echo_run"] >= echo_run_min -> decline (the assistant hearing itself)
  2. exact layer    a declared template phrase with a unique owner
  3. classifier     char-ngram TF-IDF + logistic regression; __decline__ class; confidence threshold
  4. prefix         slot-free template (>= N literal words) + trailing speech (off by default)
  5. LLM            shortlisted manifest tools + addressee gate. Low-confidence rows always go; rows the decline
                    CLASS won go per `route_decline_class`: never / active (assistant spoke within active_s) / always

`route_fast` runs 1-4 and marks rows that need the LLM; `route_llm` runs 5. The plugin calls them from different
pipeline tiers; `route` composes both (the harness reproduces its measured predictions through it).
"""
from dataclasses import dataclass

from ovos_switchboard.classifier import Classifier
from ovos_switchboard.device import DECLINE
from ovos_switchboard.exact import ExactLayer
from ovos_switchboard.labels import LabelSelector
from ovos_switchboard.schema import UNMAPPED
from ovos_switchboard.shortlist import Shortlist
from ovos_switchboard.slots import SlotExtractor
from ovos_switchboard.tools import META_PREFIX

# What each meta tool means as a canonical intent; None = no intent (decline or nothing to run).
META_CANON = {"ask_knowledge": "knowledge.ask", "stop": "stop", "unsupported": None, "ignore": None}
DECLINE_POLICIES = ("never", "active", "always")


@dataclass
class RouterConfig:
    threshold: float = 0.5
    prefix_min_literal: int = 0
    route_decline_class: str = "active"   # upstream default; "always" = full routing
    gate: float = 0.9
    shortlist_k: int = 12
    device_classes_only: bool = True
    echo_run_min: int = 4
    active_s: float = 120

    def __post_init__(self):
        if self.route_decline_class not in DECLINE_POLICIES:
            raise ValueError(f"route_decline_class must be one of {DECLINE_POLICIES}, got {self.route_decline_class!r}")


class Router:
    def __init__(self, device, schema, tools_doc, llm=None, config=None):
        """tools_doc: build_tools(device.definitions) or the harness's tools.json. llm: an LLMClient or None."""
        self.cfg = c = config or RouterConfig()
        self.schema, self.tools_doc, self.llm = schema, tools_doc, llm
        self.classifier = Classifier(device, schema, c.device_classes_only)
        self.exact = ExactLayer(device, schema)
        if c.prefix_min_literal:
            self.exact.enable_prefix(c.prefix_min_literal)
        self.labels = LabelSelector(device, schema)
        self.slots = SlotExtractor(device, schema)
        self.shortlist = Shortlist(tools_doc, device.examples) if llm else None

    def route(self, utterance, context=None):
        """context: {"gap_s": seconds since the assistant last spoke or None, "echo_run": int}.
        Returns the decision plus, for dispatch, `label` (skill_id:intent_name) and `match_data`."""
        d = self.route_fast(utterance, context)
        return self.route_llm(utterance, d) if d.get("needs_llm") else d

    def route_fast(self, utterance, context=None):
        return self._with_dispatch(self._fast(utterance, context or {}), utterance)

    def route_llm(self, utterance, decision):
        """Raises LLMUnavailable if the endpoint fails."""
        return self._with_dispatch(self._llm(utterance, decision), utterance)

    def _fast(self, utterance, context):
        utt = utterance.lower()
        if context.get("echo_run", 0) >= self.cfg.echo_run_min:
            return {"intent": None, "stage": "echo"}
        owner = self.exact.route(utt)
        if owner:
            return {"intent": owner, "stage": "exact", "confidence": 1.0,
                    "slots": self.slots.extract(owner, utt, loose=False)}
        ranked = self.classifier.rank(utt)
        top, conf = ranked[0]
        decision = {"intent": top, "stage": "classifier", "confidence": round(conf, 4),
                    "top3": [c for c, _ in ranked[:3]]}
        if not (top.startswith(UNMAPPED) or top == DECLINE or conf < self.cfg.threshold):
            return {**decision, "slots": self.slots.extract(top, utt)}
        decision.update(intent=None, decline_class=top == DECLINE)
        if self.cfg.prefix_min_literal:
            prefixed = self.exact.prefix_route(utt)
            if prefixed:
                return {**decision, "intent": prefixed, "stage": "prefix", "slots": self.slots.extract(prefixed, utt)}
        if self.llm and self._llm_eligible(top == DECLINE, context):
            decision["needs_llm"] = True
        return decision

    def _llm_eligible(self, decline_class, context):
        if not decline_class:
            return True
        policy, gap = self.cfg.route_decline_class, context.get("gap_s")
        return policy == "always" or (policy == "active" and gap is not None and gap < self.cfg.active_s)

    def _llm(self, utterance, decision):
        res = self.llm.route(utterance, self.shortlist.tools(utterance, self.cfg.shortlist_k))
        intent, slots = self._tool_intent(res["tool"], res["args"])
        gate = self.llm.p_addressed(utterance)
        out = {**{k: v for k, v in decision.items() if k != "needs_llm"},
               "intent": intent, "slots": slots, "stage": "llm", "llm_tool": res["tool"], "tool_args": res["args"],
               "p_addressed": round(gate["p_yes"], 4), "llm_ms": res["ms"] + gate["ms"]}
        if gate["p_yes"] < self.cfg.gate:
            out.update(intent=None, slots={}, stage="llm_gate")
        return out

    def _tool_intent(self, tool, args):
        """(canonical intent or None, canonical slots) for a tool call."""
        strings = {k: v for k, v in args.items() if isinstance(v, str)}
        if tool in META_CANON:
            return META_CANON[tool], strings
        label = self.tools_doc["labels"][tool]
        canon, renames = self.schema.label_map.get(label, (UNMAPPED + label, {}))
        return canon, {renames.get(k, k): v for k, v in strings.items()}

    def _with_dispatch(self, d, utterance):
        intent = d.get("intent")
        if not intent:
            return d
        if d.get("stage") in ("llm", "llm_gate") and d.get("llm_tool") in self.tools_doc["labels"]:
            label = self.tools_doc["labels"][d["llm_tool"]]
            if label.startswith(META_PREFIX):
                return d
            return {**d, "label": label, "match_data": d.get("tool_args") or self._keyword_data(label, utterance)}
        if intent.startswith(UNMAPPED):
            return {**d, "label": intent.removeprefix(UNMAPPED), "match_data": {}}
        label, data, how = self.labels.select(intent, utterance)
        return {**d, "label": label, "match_data": data, "label_how": how}

    def _keyword_data(self, label, utterance):
        """Keyword tools carry no parameters; the vocab the skill reads (message.data["Location"]) comes from
        evaluating its rule on the utterance."""
        rule = self.labels.adapt.get(label)
        hit = rule.evaluate(utterance.lower().strip()) if rule else None
        return hit[1] if hit else {}
