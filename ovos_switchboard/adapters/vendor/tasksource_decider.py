# Vendored from https://huggingface.co/tasksource/tasksource-decider-nano (tag v4), Apache-2.0, by tasksource.
# Unmodified. PyTorch + transformers only; loads safetensors, no remote code. Install with the "decider" extra.
# pylint: skip-file
# ruff: noqa
"""Self-contained inference code for a joint typed-decision cross-encoder (ModernBERT-family encoder).

Each option of a question is read as one tokenizer pair

    (state, "Question: <instructions>\\nOption: <option>")

and scored by a linear head on the attention-masked mean of the last hidden states; a question's
probabilities are the softmax over its options' scores. Nothing is truncated: a pair longer than
`max_length` (default 8,192 tokens, the encoder's native context) raises `TooLong`. All the
(question, option) pairs of a request are sorted by length and run together in padded batches.

Layout "shared" (decider_config.json) reads exactly the same tokens, but writes the state once per
request: [CLS] state [SEP], then each "Question: <instructions>\nOption:" block once, then each
" <option> [SEP]" block, with tree attention (state sees state; a question block sees the state and
itself; an option block sees the state, its question and itself, never other options). RoPE
positions follow the virtual pair, so the per-pair limit is unchanged. Pooling "opt" averages the
option block only.

    from decider import Decider
    model = Decider.from_pretrained("<repo or directory>")
    model.answer(state={"text": "..."}, questions={"q1": {"type": "choice", "instructions": "...",
                                                          "criteria": {"a": "...", "b": "..."}}})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FILES = ["config.json", "decider_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json"]


class TooLong(ValueError):
    """An input does not fit the model's context; inputs are never truncated."""


def text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def state_text(state: Any) -> str:
    return text(state) if state not in ("", None, {}, []) else "(empty)"


def format_option(key: str, desc: Any) -> str:
    desc_str = text(desc).strip() if desc is not None else ""
    key_str = str(key).strip()
    if not desc_str:
        return key_str
    is_dummy_key = (key_str.lower().startswith(("option_", "k_", "choice_", "item_"))
                    or (len(key_str) == 1 and key_str.isalpha())
                    or (key_str.startswith("(") and key_str.endswith(")") and len(key_str) <= 4))
    if is_dummy_key:
        return desc_str
    if desc_str.lower() == key_str.lower():
        return key_str
    return f"{key_str}: {desc_str}"


def question_options(q: dict) -> Tuple[List[str], List[str]]:
    """(answer keys, rendered option texts) for a choice or noul question."""
    crit = q.get("criteria", {}) or {}
    if q.get("type", "choice") == "choice":
        keys = list(crit)
        return keys, [format_option(k, crit[k]) for k in keys]
    descs = (crit.get("false", crit.get("False", "False")), crit.get("true", crit.get("True", "True")))
    return ["false", "true"], [format_option("false", descs[0]), format_option("true", descs[1])]


def tail_text(question: str, option: str) -> str:
    return f"Question: {question}\nOption: {option}"


def _resolve(path_or_repo: str, revision: Optional[str] = None) -> Path:
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path_or_repo, allow_patterns=FILES, revision=revision))


class Decider:
    def __init__(self, directory: Path, device: Optional[str] = None, max_length: Optional[int] = None,
                 token_budget: int = 65536):
        import torch
        import torch.nn as nn
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        self.config = json.loads((directory / "decider_config.json").read_text())
        self.max_length = int(max_length or self.config.get("inference_max_length", 8192))
        self.token_budget = int(token_budget)
        self.tok = AutoTokenizer.from_pretrained(str(directory))
        cfg = AutoConfig.from_pretrained(str(directory))
        net = nn.Module()
        net.backbone = AutoModel.from_config(cfg)
        net.drop = nn.Dropout(0.1)
        net.scorer = nn.Linear(cfg.hidden_size, 1)
        net.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = net.to(self.device).eval()
        self.layout = self.config.get("layout", "pair")
        self.pool = self.config.get("pool", "opt")
        self.attn = self.config.get("attn", "cached")  # shared layout: "cached" | "full" | "auto"
        self.window = getattr(cfg, "sliding_window", None)

    @classmethod
    def from_pretrained(cls, path_or_repo: str, revision: Optional[str] = None, **kw) -> "Decider":
        return cls(_resolve(path_or_repo, revision), **kw)

    def score_many(self, items: Sequence[Tuple[str, str, Sequence[str]]]) -> List[np.ndarray]:
        """items: [(state, instructions, option texts)] -> one logit array per item."""
        if self.layout == "shared":
            return self._score_shared(items)
        torch = self.torch
        pairs = [self.tok(s, tail_text(q, o), truncation=False)["input_ids"] for s, q, opts in items for o in opts]
        longest = max(map(len, pairs))
        if longest > self.max_length:
            raise TooLong(f"a (state, question, option) pair needs {longest} tokens > {self.max_length}")
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]))
        scores = np.zeros(len(pairs), dtype=np.float64)
        i = 0
        with torch.no_grad():
            while i < len(order):
                j = i
                while j < len(order) and (j - i + 1) * len(pairs[order[j]]) <= max(self.token_budget, len(pairs[order[i]])):
                    j += 1
                idx = order[i:j]
                width = len(pairs[idx[-1]])
                ids = torch.full((len(idx), width), self.tok.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                for r, k in enumerate(idx):
                    ids[r, :len(pairs[k])] = torch.tensor(pairs[k])
                    mask[r, :len(pairs[k])] = 1
                ids, mask = ids.to(self.device), mask.to(self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    h = self.net.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
                    m = mask.unsqueeze(-1).to(h.dtype)
                    z = self.net.scorer((h * m).sum(1) / m.sum(1).clamp_min(1)).squeeze(-1)
                scores[idx] = z.float().cpu().numpy()
                i = j
        out, start = [], 0
        for _, _, opts in items:
            out.append(scores[start:start + len(opts)])
            start += len(opts)
        return out

    def _full(self, n_questions: int, state_len: int, auto_extra: int = 4096) -> bool:
        """cached: the state reads only itself and is shared by all questions of a request; full: one row per
        question, the state also reads its question; auto: full when that costs <= auto_extra extra tokens."""
        if self.attn == "auto":
            return (n_questions - 1) * state_len <= auto_extra
        return self.attn == "full"

    def _score_shared(self, items, row_extra: int = 2048) -> List[np.ndarray]:
        torch, tok = self.torch, self.tok
        rows, oid, prefix = [], 0, {}
        runs = []  # questions per request = consecutive items with the same state
        for s, _, _ in items:
            if runs and runs[-1][0] == s:
                runs[-1][1] += 1
            else:
                runs.append([s, 1])
        nq = [n for _, n in runs for _ in range(n)]
        first = [k == 0 for _, n in runs for k in range(n)]
        for (s, q, opts), n_q, new_req in zip(items, nq, first):  # rows of max(2P, P + row_extra) tokens
            p = prefix.setdefault(s, tok(s)["input_ids"])
            full = self._full(n_q, len(p))
            qi = tok(f"Question: {q}\nOption:", add_special_tokens=False)["input_ids"]
            ob = [tok(" " + o, add_special_tokens=False)["input_ids"] + [tok.sep_token_id] for o in opts]
            longest = len(p) + len(qi) + max(map(len, ob))
            if longest > self.max_length:
                raise TooLong(f"a (state, question, option) pair needs {longest} tokens > {self.max_length}")
            cap = max(2 * len(p), len(p) + row_extra)
            if not rows or rows[-1][0] is not p or new_req or full or rows[-1][2] + len(qi) + len(ob[0]) > cap:
                rows.append([p, [], len(p), full])
            k = 0
            while k < len(ob):
                row = rows[-1]
                take, size = [], len(qi)
                while k + len(take) < len(ob) and (not take or row[2] + size + len(ob[k + len(take)]) <= cap):
                    take.append((oid + k + len(take), ob[k + len(take)]))
                    size += len(take[-1][1])
                if row[1] and row[2] + size > cap:
                    rows.append([p, [], len(p), full])
                    continue
                row[1].append((qi, take))
                row[2] += size
                k += len(take)
            oid += len(ob)
        scores = np.zeros(oid, dtype=np.float64)
        order = sorted(range(len(rows)), key=lambda i: rows[i][2])
        i = 0
        with torch.no_grad():
            while i < len(order):
                j = i + 1
                while (j < len(order) and (j - i + 1) * rows[order[j]][2] <= self.token_budget
                       and (j - i + 1) * rows[order[j]][2] ** 2 <= 2 ** 26):  # bound dense mask size
                    j += 1
                batch = [rows[k] for k in order[i:j]]
                L, B = max(r[2] for r in batch), len(batch)
                ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
                pos = torch.zeros((B, L), dtype=torch.long)
                kind = torch.full((B, L), -1, dtype=torch.long)
                qix = torch.full((B, L), -1, dtype=torch.long)
                oix = torch.full((B, L), -1, dtype=torch.long)
                spans = []
                rowfull = torch.tensor([r[3] for r in batch], dtype=torch.bool)
                for r, (p, blocks, _, _) in enumerate(batch):
                    ids[r, :len(p)], pos[r, :len(p)], kind[r, :len(p)] = torch.tensor(p), torch.arange(len(p)), 0
                    c = len(p)
                    for b, (qi, take) in enumerate(blocks):
                        n = len(qi)
                        ids[r, c:c + n], pos[r, c:c + n] = torch.tensor(qi), torch.arange(len(p), len(p) + n)
                        kind[r, c:c + n], qix[r, c:c + n] = 1, b
                        c += n
                        for o, x in take:
                            m = len(x)
                            ids[r, c:c + m] = torch.tensor(x)
                            pos[r, c:c + m] = torch.arange(len(p) + n, len(p) + n + m)
                            kind[r, c:c + m], qix[r, c:c + m], oix[r, c:c + m] = 2, b, o
                            spans.append((o, r, c, m, len(p), (r, len(p) + sum(len(bb[0]) + sum(len(y) for _, y in bb[1]) for bb in blocks[:b])), n))
                            c += m
                ids, pos, kind, qix, oix, rowfull = (t.to(self.device) for t in (ids, pos, kind, qix, oix, rowfull))
                ki, kj = kind[:, :, None], kind[:, None, :]
                allow = (kj == 0) & (ki >= 0)
                allow |= (kj == 1) & (ki >= 1) & (qix[:, :, None] == qix[:, None, :])
                allow |= (kj == 2) & (ki == 2) & (oix[:, :, None] == oix[:, None, :])
                allow |= rowfull[:, None, None] & (ki == 0) & (kj == 1)  # full rows: the state reads its question
                allow |= torch.eye(L, dtype=torch.bool, device=self.device)[None]
                masks = {"full_attention": allow[:, None]}
                if self.window is not None:
                    masks["sliding_attention"] = (allow & ((pos[:, :, None] - pos[:, None, :]).abs() <= self.window))[:, None]
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    h = self.net.backbone(input_ids=ids, attention_mask=masks, position_ids=pos).last_hidden_state
                h = h.float()
                if self.pool == "opt":  # vectorized mean over each option block
                    sel = oix >= 0
                    flat = oix[sel]
                    uniq, inv = torch.unique(flat, return_inverse=True)
                    v = torch.zeros(len(uniq), h.shape[-1], device=h.device).index_add_(0, inv, h[sel])
                    v = v / torch.bincount(inv).unsqueeze(-1).to(v.dtype)
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                        z = self.net.scorer(v).squeeze(-1)
                    scores[uniq.cpu().numpy()] = z.float().cpu().numpy()
                    i = j
                    continue
                vecs = []
                for o, r, c, m, plen, (_, qstart), n in spans:
                    v = h[r, c:c + m].sum(0)
                    if self.pool == "pair":
                        v = (v + h[r, :plen].sum(0) + h[r, qstart:qstart + n].sum(0)) / (plen + n + m)
                    else:
                        v = v / m
                    vecs.append(v)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    z = self.net.scorer(torch.stack(vecs)).squeeze(-1)
                scores[[sp[0] for sp in spans]] = z.float().cpu().numpy()
                i = j
        out, start = [], 0
        for _, _, opts in items:
            out.append(scores[start:start + len(opts)])
            start += len(opts)
        return out

    def answer(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Decision Index request -> {"answers": {qid: answer}}. Supports `choice` and `noul`."""
        s = state_text(state)
        parsed = []
        for qid, q in questions.items():
            kind = q.get("type", "choice")
            if kind not in ("choice", "noul") or (kind == "choice" and not q.get("criteria")):
                raise ValueError(f"unsupported question type {kind!r}")
            keys, opts = question_options(q)
            parsed.append((qid, kind, keys, opts, text(q.get("instructions", ""))))
        logits = self.score_many([(s, ins, opts) for _, _, _, opts, ins in parsed])
        answers = {}
        for (qid, kind, keys, _, _), z in zip(parsed, logits):
            p = np.exp(z - z.max())
            p /= p.sum()
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": float(np.clip(p[1], 0.0, 1.0))}
            else:
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {k: float(v) for k, v in zip(keys, p)}}
        return {"answers": answers}
