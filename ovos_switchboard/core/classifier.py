"""Stage 3: char-ngram TF-IDF + logistic regression over canonical intents, with a `__decline__` class.

Training data, in this order (order matters for bit-exact reproduction of the measured results):
  1. the device's own phrases (template expansions, keyword examples), labelled canonical or __unmapped__:<label>
  2. one pseudo-example per canonical intent: its schema description
  3. reviewed synthetic examples (slot values masked) and decline negatives
  4. other devices' phrases for intents this device can run (optional)
Then filtered to classes this device can execute: its own canonical intents, its own unmapped labels,
knowledge.ask and stop (served by pipelines, not intents), and __decline__.
"""
import hashlib
import inspect
import re
import sys

import numpy as np

from ovos_switchboard.core.device import DECLINE
from ovos_switchboard.core.schema import UNMAPPED
from ovos_switchboard.core.templates import SLOT_TOKEN


PIPELINE_SERVED = {"knowledge.ask", "stop"}
LOGREG_C = 10.0
LOGREG_MAX_ITER = 3000


def _vectorizers():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-z0-9']+")]


def _logreg():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER, class_weight="balanced")


class CharEncoder:
    """Character and word n-gram TF-IDF features."""
    def __init__(self, vectorizers=None):
        from sklearn.pipeline import make_union
        self.vectorizers = vectorizers or _vectorizers()
        self.v = make_union(*self.vectorizers)

    def fit(self, texts):
        """Learn the vocabulary from the training texts."""
        self.v.fit(texts)
        self.vectorizers = [vec for _, vec in self.v.transformer_list]
        return self

    def encode(self, texts):
        """L2-normalized features for `texts`."""
        from sklearn.preprocessing import normalize
        return normalize(self.v.transform(texts))

    def state(self):
        """(arrays, meta): the fitted vocabularies and IDF weights as plain data."""
        arrays = {f"idf_{i}": vec.idf_ for i, vec in enumerate(self.vectorizers)}
        vocab = [sorted(vec.vocabulary_, key=vec.vocabulary_.get) for vec in self.vectorizers]
        return arrays, {"vocab": vocab}

    @classmethod
    def from_state(cls, arrays, meta):
        """The encoder state() describes, without refitting."""
        vectorizers = _vectorizers()
        for i, (vec, terms) in enumerate(zip(vectorizers, meta["vocab"])):
            vec.vocabulary_ = {term: j for j, term in enumerate(terms)}
            vec.idf_ = arrays[f"idf_{i}"]
        return cls(vectorizers)


class LogRegHead:
    """Class-balanced logistic regression over the encoded features."""
    def fit(self, features, labels):
        """Train on encoded features and their labels."""
        self.m = _logreg().fit(features, labels)
        self.classes = list(self.m.classes_)
        return self

    def scores(self, features):
        """Class probabilities per row, in `classes` order."""
        return self.m.predict_proba(features)

    def state(self):
        """(arrays, meta): the fitted coefficients and classes as plain data."""
        return {"coef": self.m.coef_, "intercept": self.m.intercept_}, {"classes": self.classes}

    @classmethod
    def from_state(cls, arrays, meta):
        """The head state() describes, without refitting."""
        head = cls()
        head.m = _logreg()
        head.m.classes_ = np.array(meta["classes"])
        head.m.coef_, head.m.intercept_ = arrays["coef"], arrays["intercept"]
        head.classes = list(meta["classes"])
        return head


def phrase_rows(examples, schema):
    """(text, canonical-or-unmapped label) per example, slots stripped."""
    out = []
    for e in examples:
        t = re.sub(r"\s+", " ", SLOT_TOKEN.sub(" ", e["example"]).strip())
        if t:
            out.append((t.lower(), schema.canon_or_unmapped(e["label"])))
    return out


def training_data(device, schema, synthetic=(), device_classes_only=True):
    """(text, label) rows: device phrases, schema descriptions and synthetic rows. With device_classes_only, only
    classes this device can execute (plus the decline class)."""
    rows = phrase_rows(device.examples, schema)
    rows += [(spec["description"].lower(), canon) for canon, spec in schema.intents.items()]
    rows += [(r.get("masked", r["example"]).lower(), r["intent"]) for r in synthetic]
    if not device_classes_only:
        return rows
    rows += phrase_rows(device.shared_examples, schema)
    own = set(device.own_labels())
    own_unmapped = {UNMAPPED + label for label in own if label not in schema.label_map}
    executable = {schema.label_map[label][0] for label in own
                  if label in schema.label_map and schema.label_map[label][0]} | PIPELINE_SERVED
    return [(t, label) for t, label in rows
            if label == DECLINE or label in executable or (label.startswith(UNMAPPED) and label in own_unmapped)]


def model_fingerprint():
    """Everything besides the training rows that determines the trained model: this module's code and the
    numeric libraries' versions. A model cache keys on this plus the rows."""
    import sklearn
    code = hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode()).hexdigest()
    return {"code": code, "sklearn": sklearn.__version__, "numpy": np.__version__}


class Classifier:
    """Intent classifier trained on the device's own phrases plus reviewed synthetic examples."""
    def __init__(self, encoder, head):
        self.encoder, self.head, self.classes = encoder, head, head.classes

    @classmethod
    def train(cls, rows):
        """Trained on (text, label) pairs, as training_data() returns them."""
        texts, labels = [t for t, _ in rows], [label for _, label in rows]
        encoder = CharEncoder().fit(texts)
        return cls(encoder, LogRegHead().fit(encoder.encode(texts), labels))

    def state(self):
        """(arrays, meta): the trained model as numpy arrays plus JSON-able metadata. No pickle: loading it can't
        run code."""
        enc_arrays, enc_meta = self.encoder.state()
        head_arrays, head_meta = self.head.state()
        return {**enc_arrays, **head_arrays}, {**enc_meta, **head_meta}

    @classmethod
    def from_state(cls, arrays, meta):
        """The classifier state() describes, without retraining."""
        return cls(CharEncoder.from_state(arrays, meta), LogRegHead.from_state(arrays, meta))

    @classmethod
    def for_device(cls, device, schema, synthetic=(), device_classes_only=True):
        """synthetic: rows carrying their canonical intent or DECLINE (reviewed examples, dialog negatives)."""
        return cls.train(training_data(device, schema, synthetic, device_classes_only))

    def rank(self, utterance):
        """[(class, probability)] best first."""
        probs = self.head.scores(self.encoder.encode([utterance]))[0]
        return [(self.classes[j], float(probs[j])) for j in np.argsort(-probs)]
