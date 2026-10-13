"""Pick the K tools most similar to an utterance, so the LLM prompt stays small.

Every manifest tool at once overflows a modest model's context (Mac: 149 tools, ~12.1k tokens), so a shortlist is
required, not an optimization. Similarity = max char-ngram cosine to any of a tool's phrases or its description.
Meta tools are always appended. Deterministic: same inputs, same shortlist.
"""
import re

import numpy as np

from ovos_switchboard.templates import SLOT_TOKEN
from ovos_switchboard.tools import META_PREFIX

DEFAULT_K = 12


class Shortlist:
    def __init__(self, tools_doc, examples):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.by_name = {t["function"]["name"]: t for t in tools_doc["tools"]}
        self.meta = [n for n, label in tools_doc["labels"].items() if label.startswith(META_PREFIX)]
        label2name = {label: n for n, label in tools_doc["labels"].items()}
        texts, owners = [], []
        for e in examples:
            if e["label"] in label2name:
                texts.append(re.sub(r"\s+", " ", SLOT_TOKEN.sub(" ", e["example"])).strip().lower())
                owners.append(label2name[e["label"]])
        for n, t in self.by_name.items():
            if n not in self.meta:
                texts.append(t["function"]["description"].lower())
                owners.append(n)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(texts)
        self.X = self.vec.transform(texts)
        self.owners = np.array(owners)
        self.idx = {n: np.where(self.owners == n)[0] for n in sorted(set(owners))}

    def rank(self, utterance):
        sims = (self.X @ self.vec.transform([utterance.lower()]).T).toarray().ravel()
        best = {n: float(sims[i].max()) for n, i in self.idx.items()}
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))

    def tools(self, utterance, k=DEFAULT_K):
        top = [n for n, _ in self.rank(utterance)[:k]]
        return [self.by_name[n] for n in top + self.meta]
