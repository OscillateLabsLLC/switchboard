"""Stage 3: char-ngram TF-IDF + logistic regression over canonical intents, with a `__decline__` class.

Training data, in this order (order matters for bit-exact reproduction of the measured results):
  1. the device's own phrases (template expansions, keyword examples), labelled canonical or __unmapped__:<label>
  2. one pseudo-example per canonical intent: its schema description
  3. reviewed synthetic examples (slot values masked) and decline negatives
  4. other devices' phrases for intents this device can run (optional)
Then filtered to classes this device can execute: its own canonical intents, its own unmapped labels,
knowledge.ask and stop (served by pipelines, not intents), and __decline__.
"""
import re

import numpy as np

from ovos_switchboard.device import DECLINE
from ovos_switchboard.schema import UNMAPPED
from ovos_switchboard.templates import SLOT_TOKEN

PIPELINE_SERVED = {"knowledge.ask", "stop"}
LOGREG_C = 10.0
LOGREG_MAX_ITER = 3000


class CharEncoder:
    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_union
        self.v = make_union(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z0-9']+"),
        ).fit(texts)
        return self

    def encode(self, texts):
        from sklearn.preprocessing import normalize
        return normalize(self.v.transform(texts))


class LogRegHead:
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER, class_weight="balanced").fit(X, y)
        self.classes = list(self.m.classes_)
        return self

    def scores(self, X):
        return self.m.predict_proba(X)


def phrase_rows(examples, schema):
    """(text, canonical-or-unmapped label) per example, slots stripped."""
    out = []
    for e in examples:
        t = re.sub(r"\s+", " ", SLOT_TOKEN.sub(" ", e["example"]).strip())
        if t:
            out.append((t.lower(), schema.canon_or_unmapped(e["label"])))
    return out


def training_data(device, schema, device_classes_only=True):
    rows = phrase_rows(device.examples, schema)
    rows += [(spec["description"].lower(), canon) for canon, spec in schema.intents.items()]
    rows += [(r.get("masked", r["example"]).lower(), r["intent"]) for r in device.synthetic]
    if not device_classes_only:
        return rows
    rows += phrase_rows(device.shared_examples, schema)
    own = set(device.own_labels())
    own_unmapped = {UNMAPPED + label for label in own if label not in schema.label_map}
    executable = {schema.label_map[label][0] for label in own
                  if label in schema.label_map and schema.label_map[label][0]} | PIPELINE_SERVED
    return [(t, label) for t, label in rows
            if label == DECLINE or label in executable or (label.startswith(UNMAPPED) and label in own_unmapped)]


class Classifier:
    def __init__(self, device, schema, device_classes_only=True):
        rows = training_data(device, schema, device_classes_only)
        texts, labels = [t for t, _ in rows], [label for _, label in rows]
        self.encoder = CharEncoder().fit(texts)
        self.head = LogRegHead().fit(self.encoder.encode(texts), labels)
        self.classes = self.head.classes

    def rank(self, utterance):
        """[(class, probability)] best first."""
        probs = self.head.scores(self.encoder.encode([utterance]))[0]
        return [(self.classes[j], float(probs[j])) for j in np.argsort(-probs)]
