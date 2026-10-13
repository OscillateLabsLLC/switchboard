"""Raw slot strings from the skills' own templates, under canonical slot names (used for reporting and the
decision record; dispatch uses the label selector's skill-native match_data).

The exact layer already matched a template like "turn on {entity}"; the slot value is the text the wildcard covered.
For classifier picks the predicted intent's templates are tried LOOSE (extra words allowed before the template), so
"could you turn on the desk lamp please" still yields entity=desk lamp. Entity resolution and typed parsing stay in
the skills.

Template slot names map to canonical slots via per-label renames; failing that, a template slot whose name is a
canonical slot of the intent is kept; failing that, if the intent has exactly one canonical slot and the template one
slot, they correspond. Unmappable slots are dropped (never guessed).
"""
import re

from ovos_switchboard.core.templates import LITERAL_EXPANSION_LIMIT, SLOT, literal_words, named_template_regex


TRAILING_COURTESY = re.compile(r"\s+(please|thanks|thank you)$")


class SlotExtractor:
    """Canonical slot values from the device's templates for a canonical intent."""
    def __init__(self, device, schema):
        """Other devices' manifests contribute templates too (Neon's Adapt-only alerts have none; the OVOS alerts
        templates describe the same canonical intents)."""
        self.schema = schema
        self.by_canon = {}
        for definitions in (device.definitions, *device.shared_definitions):
            for skill_id, defs in definitions.items():
                for d in defs if isinstance(defs, list) else []:
                    if d.get("method") == "template":
                        self._add(f"{skill_id}:{d['intent_name']}", d["definition"].get("samples") or [])

    def _add(self, label, samples):
        canon, renames = self.schema.label_map.get(label, (None, {}))
        for line in samples:
            if not SLOT.search(line):
                continue
            try:
                body = named_template_regex(line)
                full = re.compile(r"^\s*" + body + r"\s*[?.!]*$")
                # loose: extra words allowed BEFORE the template; the end stays anchored, so a trailing slot runs to
                # the end (courtesy words are trimmed in _canonical)
                loose = re.compile(r"(?:^|\s)" + body + r"\s*[?.!]*$")
            except (ValueError, re.error):
                continue
            if canon:
                self.by_canon.setdefault(canon, []).append(
                    (literal_words(line, LITERAL_EXPANSION_LIMIT), full, loose, renames))

    def extract(self, canon, utterance, loose=True):
        """Canonical slots for `canon` from the best-matching template (most literal words), or {}."""
        utt = utterance.lower().strip()
        for mode in ("full", "loose") if loose else ("full",):
            hits = [(lit, m, renames) for lit, full, lrx, renames in self.by_canon.get(canon, [])
                    if (m := full.match(utt) if mode == "full" else lrx.search(utt))]
            if hits:
                _, match, renames = max(hits, key=lambda candidate: candidate[0])  # ties: the first template wins
                return self._canonical(canon, match, renames)
        return {}

    def _canonical(self, canon, match, renames):
        slots, allowed, raw = {}, self.schema.canon_slots.get(canon, []), {}
        for group, value in match.groupdict().items():
            value = TRAILING_COURTESY.sub("", (value or "").strip())
            if value:
                raw.setdefault(group.split("__")[0], value)
        for name, value in raw.items():
            target = renames.get(name) or (name if name in allowed else None)
            if target is None and len(allowed) == 1 and len(raw) == 1:
                target = allowed[0]
            if target:
                slots[target] = value
        return slots
