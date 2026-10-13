"""The canonical intent schema (intents.yaml): shared intent names, their slots, risk levels, and which skill labels
(`skill_id:intent_name`) serve each one. Canonical intents let devices with different skills share training data;
dispatch always goes back to a label the device actually registered."""
from importlib import resources

import yaml

UNMAPPED = "__unmapped__:"


class Schema:
    def __init__(self, doc):
        self.intents = doc["intents"]
        self.declined = doc.get("declined") or []
        # label -> canonical, ovos entries only (declined labels stay unmapped competitors)
        self.label2canon = {}
        # label -> (canonical or None, {skill_slot: canonical_slot}); declined labels map to None
        self.label_map = {label: (None, {}) for label in self.declined}
        for canon, spec in self.intents.items():
            for e in spec.get("ovos") or []:
                label, renames = (e, {}) if isinstance(e, str) else (e["label"], e.get("slots") or {})
                self.label2canon[label] = canon
                self.label_map[label] = (canon, renames)
                self.label_map[label + ".intent"] = (canon, renames)  # legacy topic name vs INTENT-4 canonical
        self.canon_slots = {c: spec.get("slots") or [] for c, spec in self.intents.items()}

    @classmethod
    def load(cls, path=None):
        """`path`, or the schema bundled with the package."""
        if path:
            with open(path) as f:
                return cls(yaml.safe_load(f))
        return cls(yaml.safe_load(resources.files("ovos_switchboard.data").joinpath("intents.yaml").read_text()))

    def canon_or_unmapped(self, label):
        return self.label2canon.get(label, UNMAPPED + label)

    def risk(self, canon):
        return (self.intents.get(canon) or {}).get("risk", "low")
