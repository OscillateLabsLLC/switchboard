"""Pick the device's SKILL LABEL for a canonical intent, and build the skill-native match_data.

Dispatch needs `skill_id:intent_name` plus the data that skill reads; canonical intents are for sharing and scoring.
Many canonical intents have several labels on a device (Hub: settings.change has 29), so choosing among them is part
of routing. Uses only THIS device's declarations:
  Adapt labels      evaluate the rule like Adapt: every `require` vocab present (whole word), each `one_of`
                    satisfied; score = required + optional vocab matched. match_data = {VocabName: matched text}, the
                    keys Adapt skills read (message.data["Location"]); regex entities fill named groups.
  template labels   template match (full, then loose); score = literal words; match_data = slot values under the
                    template's own slot names.
  neither matches   char-ngram similarity to the label's examples, no match_data.
Deterministic.
"""
import re

from ovos_switchboard.device import vkey
from ovos_switchboard.templates import SLOT_TOKEN, compiles, literal_words, named_template_regex

LITERAL_EXPANSION_LIMIT = 50
EXAMPLES_PER_LABEL = 200


def phrase_regex(phrases):
    alts = sorted({re.escape(p.lower()) for p in phrases if p.strip()}, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(alts) + r")\b") if alts else None


class AdaptRule:
    def __init__(self, definition):
        self.steps = definition["steps"]
        self.vocab = {k: phrase_regex(v) for k, v in (definition.get("vocab") or {}).items()}
        self.regex = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in (definition.get("regex") or {}).items()}

    def _find(self, name, utt):
        key = vkey(name)
        rx = self.vocab.get(key)
        if rx:
            m = rx.search(utt)
            if m:
                return m.group(1)
        for pat in self.regex.get(key, []):
            m = pat.search(utt)
            if m:
                return next((v for g, v in m.groupdict().items() if vkey(g) == key and v), None)
        return None

    def evaluate(self, utt):
        """(score, match_data) if the rule fires, else None."""
        data, score = {}, 0
        for method, args in self.steps:
            if method == "require":
                v = self._find(args[0], utt)
                if v is None:
                    return None
                data[args[0]], score = v, score + 2
            elif method == "one_of":
                hits = [(a, v) for a, v in ((a, self._find(a, utt)) for a in args) if v is not None]
                if not hits:
                    return None
                data[hits[0][0]], score = hits[0][1], score + 2
            elif method == "optionally":
                v = self._find(args[0], utt)
                if v is not None:
                    data[args[0]], score = v, score + 1
        return score, data


class LabelSelector:
    def __init__(self, device, schema):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.by_canon, self.adapt, self.templates, ex_text = {}, {}, {}, {}
        for skill_id, defs in device.definitions.items():
            for d in defs if isinstance(defs, list) else []:
                label = f"{skill_id}:{d['intent_name']}"
                canon = schema.label_map.get(label, (None,))[0]
                if not canon:
                    continue
                self.by_canon.setdefault(canon, []).append(label)
                if d["method"] == "adapt":
                    self.adapt[label] = AdaptRule(d["definition"])
                else:
                    self.templates[label] = [self._template(line) for line in d["definition"].get("samples") or []
                                             if compiles(line)]
        for e in device.examples:
            ex_text.setdefault(e["label"], []).append(SLOT_TOKEN.sub(" ", e["example"]).lower())
        self.ex_text = {label: " ".join(v[:EXAMPLES_PER_LABEL]) for label, v in ex_text.items()}
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(list(self.ex_text.values()) or ["x"])

    @staticmethod
    def _template(line):
        body = named_template_regex(line)
        return (literal_words(line, LITERAL_EXPANSION_LIMIT), re.compile(r"^\s*" + body + r"\s*[?.!]*$"),
                re.compile(r"(?:^|\s)" + body + r"\s*[?.!]*$"))

    def select(self, canon, utterance):
        """(label or None, match_data, how)."""
        labels = self.by_canon.get(canon, [])
        if not labels:
            return None, {}, "no label"
        utt = utterance.lower().strip()
        scored = [s for s in (self._score(label, utt) for label in labels) if s]
        if scored:
            # ties: the more specific match wins (fewer slots: "tell me a joke" beats "tell me a joke [about {q}]")
            _, label, data, how = max(scored, key=lambda s: (s[0], -len(s[2]), s[1]))
            return label, data, how
        if len(labels) == 1:
            return labels[0], {}, "only label"
        q = self.vec.transform([utt])
        sims = [(float((self.vec.transform([self.ex_text.get(label, "")]) @ q.T).toarray()[0, 0]), label)
                for label in labels]
        return max(sims)[1], {}, "nearest examples"

    def _score(self, label, utt):
        if label in self.adapt:
            hit = self.adapt[label].evaluate(utt)
            return (hit[0], label, hit[1], "adapt rule") if hit else None
        for mode in ("full", "loose"):
            best = None
            for lit, full, loose in self.templates.get(label, []):
                m = full.match(utt) if mode == "full" else loose.search(utt)
                if m and (best is None or lit > best[0]):
                    best = (lit, {g.split("__")[0]: v.strip() for g, v in m.groupdict().items() if v})
            if best:
                return best[0] + (1 if mode == "full" else 0), label, best[1], f"template {mode}"
        return None
