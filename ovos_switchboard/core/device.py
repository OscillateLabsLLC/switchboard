"""What one device can do: its intent manifest and the training phrases derived from it.

The manifest is a dict `skill_id -> [definition, ...]` in INTENT-4 describe shape. Template definitions carry
`samples`; keyword (Adapt) definitions are normalized to `method: "adapt"` with `steps`, `vocab`, `regex` and
`keyword_examples`, the shape the label selector evaluates.

Two sources, same shape: files (the intent-eval harness, for regression) and live registrations (the plugin).
"""
import itertools
import json
import re
import string
from dataclasses import dataclass, field

from ovos_switchboard.core.templates import MAX_PER_LINE, SLOT, clean, expand, expansions, template_regex

MAX_KEYWORD_EXAMPLES = 40
DECLINE = "__decline__"  # training class for "should not act"


def vkey(name):
    """Adapt vocab names vs file stems differ in case and separators ("RelativeDay" / relative-day.voc)."""
    return re.sub(r"[-_]", "", name.lower())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


@dataclass
class Device:
    definitions: dict
    examples: list                    # {"label", "example", ...}: template expansions + keyword examples
    lang: str = "en-us"
    shared_examples: list = field(default_factory=list)      # other devices' phrases (classifier only)
    shared_definitions: list = field(default_factory=list)   # other devices' manifests (slot templates only)

    def own_labels(self):
        return [f"{sid}:{d['intent_name']}" for sid, defs in self.definitions.items()
                if isinstance(defs, list) for d in defs]

    def template_entries(self, definitions=None):
        """(label, line) for every template sample in this device's language."""
        for skill_id, defs in (definitions or self.definitions).items():
            for d in defs if isinstance(defs, list) else []:
                if d["lang"].lower() == self.lang and d["method"] == "template":
                    for line in d["definition"].get("samples") or []:
                        yield f"{skill_id}:{d['intent_name']}", line

    @classmethod
    def from_files(cls, manifest, examples, lang="en-us", shared_examples=(), shared_manifests=()):
        def manifest_defs(path):
            with open(path) as f:
                return json.load(f)["definitions"]
        return cls(definitions=manifest_defs(manifest), examples=read_jsonl(examples), lang=lang,
                   shared_examples=[r for p in shared_examples for r in read_jsonl(p)],
                   shared_definitions=[manifest_defs(p) for p in shared_manifests])

    @classmethod
    def from_definitions(cls, definitions, lang="en-us"):
        """Live: derive training phrases from the manifest the way the harness did (harvest_live.py for templates,
        build_hub_manifest.py keyword combinations for Adapt)."""
        return cls(definitions=definitions, examples=derive_examples(definitions, lang), lang=lang)


def derive_examples(definitions, lang):
    out, seen = [], set()
    for skill_id, defs in definitions.items():
        for d in defs if isinstance(defs, list) else []:
            if d["lang"].lower() != lang:
                continue
            label = f"{skill_id}:{d['intent_name']}"
            if d["method"] == "template":
                rows = [(ex, sorted(set(SLOT.findall(ex))))
                        for line in d["definition"].get("samples") or [] for ex in expansions(line)]
            else:
                rows = [(ex, []) for ex in d["definition"].get("keyword_examples") or []]
            for ex, slots in rows:
                key = (label, ex.lower())
                if key not in seen:
                    seen.add(key)
                    out.append({"label": label, "skill_id": skill_id, "intent": d["intent_name"],
                                "example": ex, "slots": slots})
    return out


def keyword_examples(steps, vocab):
    """Deterministic keyword combinations: one phrase per required vocab, in builder order, plus each one_of."""
    required, choices = [], []
    for method, args in steps:
        if method == "require" and args:
            required.append(vocab.get(vkey(args[0]), [args[0].replace("_", " ")])[:6])
        elif method == "one_of":
            choices.append([p for a in args for p in vocab.get(vkey(a), [a.replace("_", " ")])[:3]])
    slots = required + choices
    if not slots:
        return []
    return [" ".join(combo) for combo in itertools.islice(itertools.product(*slots), MAX_KEYWORD_EXAMPLES)]


def adapt_definition(payload):
    """INTENT-4 §5.2 keyword registration (required / optional / one_of / excluded descriptors, samples inlined)
    -> the adapt shape: builder steps, phrases per vocab, keyword examples."""
    steps, vocab, regex = [], {}, {}

    def note(desc):
        name = desc["name"] if isinstance(desc, dict) else str(desc)
        phrases = []
        for line in (desc.get("samples") if isinstance(desc, dict) else None) or []:
            if "(?P<" in line:
                regex.setdefault(vkey(name), []).append(line)
            else:
                phrases += expansions(line, MAX_PER_LINE)
        if phrases:
            vocab.setdefault(vkey(name), list(dict.fromkeys(phrases)))
        return name

    for desc in payload.get("required") or []:
        steps.append(["require", [note(desc)]])
    for group in payload.get("one_of") or []:
        steps.append(["one_of", [note(d) for d in group]])
    for desc in payload.get("optional") or []:
        steps.append(["optionally", [note(desc)]])
    return {"steps": steps, "vocab": vocab, "regex": regex, "keyword_examples": keyword_examples(steps, vocab)}


def harvest_dialog_negatives(dialog_lines, definitions, lang, max_per_line=5):
    """Decline negatives from skills' own .dialog lines: text the assistant SPEAKS and can hear back (self-echo).
    Lines matching a template are dropped: they would teach the decline class to swallow real requests."""
    regexes = [template_regex(line) for _, line in Device(definitions, [], lang).template_entries()]
    out, seen = [], set()
    for line in dialog_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for ex in itertools.islice(expand(line), max_per_line):
            text = normalize(DIALOG_SLOT.sub(" ", clean(ex)))  # slot values unknown: drop them
            if len(text.split()) < 2 or text in seen:
                continue
            seen.add(text)
            if not any(rx.match(text) for rx in regexes):
                out.append({"intent": DECLINE, "example": text, "kind": "assistant_echo"})
    return out


DIALOG_SLOT = re.compile(r"\{+[^}]*\}+")
PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(s):
    return re.sub(r"\s+", " ", s.lower().translate(PUNCTUATION)).strip()
