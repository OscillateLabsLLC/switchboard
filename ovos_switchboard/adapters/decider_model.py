"""DeciderPort over a local typed-decision cross-encoder (default: tasksource-decider-nano, 150M, Apache-2.0, CPU).

The model reads one packed row per question, [CLS] state [SEP] "Question: <q>\\nOption:" then " <option> [SEP]"
for every option, with the attention its "shared"/"cached" layout was trained with: the state attends only to
itself, the question to the state and itself, each option to the state, the question and itself (never another
option). Each option's positions continue from the question's, as if it were alone in its own row, so it scores as
it would on its own while the state and question are encoded once. A linear head on the mean of an option's last
hidden states gives its score; probabilities are the softmax over the options.

Needs the "decider" extra (torch, transformers). The weights download from the Hugging Face Hub on first use and stay
in its cache; only config, tokenizer and safetensors files are fetched, never code.
"""
import json
import time
from dataclasses import dataclass

from ovos_switchboard.core.ports import DeciderPort


STATE_GROUP, QUESTION_GROUP = -2, -1   # option k's tokens are group k
MODEL_FILES = ["config.json", "decider_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


@dataclass(frozen=True)
class DeciderModel:
    """Which model to load. Field names are the decider step's config keys."""
    model: str = "tasksource/tasksource-decider-nano"
    revision: str = "v4"            # the measured weights
    device: str = "cpu"


def option_text(key, description):
    """The model's option rendering: "key: description" (the key alone when there's no description)."""
    description = (description or "").strip()
    return f"{key}: {description}" if description and description.lower() != key.lower() else key


def group_mask(groups, positions, window):
    """Attention allowed (query i -> key j) for one packed row: everyone sees the state; the question and the options
    see the question; an option sees only its own tokens beyond that. With a window, also |pos_i - pos_j| <= window
    (the encoder's local layers), on the per-option positions."""
    gi, gj = groups[:, None], groups[None, :]
    allow = (gj == STATE_GROUP) | ((gj == QUESTION_GROUP) & (gi != STATE_GROUP)) | ((gj >= 0) & (gi == gj))
    if window is not None:
        allow &= (positions[:, None] - positions[None, :]).abs() <= window
    return allow[None, None]


class TasksourceDecider(DeciderPort):
    """A tasksource-format joint cross-encoder answering choice questions."""

    def __init__(self, spec=DeciderModel()):
        import torch  # torch, transformers, safetensors, huggingface_hub: the "decider" extra
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        path = snapshot_download(spec.model, revision=spec.revision, allow_patterns=MODEL_FILES)
        config = AutoConfig.from_pretrained(path)
        weights = load_file(f"{path}/model.safetensors")
        self.torch, self.device = torch, torch.device(spec.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.encoder = AutoModel.from_config(config)
        self.encoder.load_state_dict({k.removeprefix("backbone."): v for k, v in weights.items()
                                      if k.startswith("backbone.")}, strict=True)
        self.head = torch.nn.Linear(config.hidden_size, 1)
        self.head.load_state_dict({"weight": weights["scorer.weight"], "bias": weights["scorer.bias"]})
        self.encoder.to(self.device).eval()
        self.head.to(self.device).eval()
        self.window = getattr(config, "sliding_window", None)

    def choose(self, state, question, options):
        t0 = time.perf_counter()
        scores = self.scores(json.dumps({"speech": state}, ensure_ascii=False, separators=(",", ":")), question,
                             [option_text(k, d) for k, d in options.items()])
        probabilities = dict(zip(options, self.torch.softmax(scores.double(), 0).tolist()))
        return {"choice": max(probabilities, key=probabilities.get), "probabilities": probabilities,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def scores(self, state, question, options):
        """One score per option, from a single packed row."""
        ids, groups, positions = self.packed_row(state, question, options)
        masks = {"full_attention": group_mask(groups, positions, None)}
        if self.window is not None:
            masks["sliding_attention"] = group_mask(groups, positions, self.window)
        with self.torch.no_grad():
            hidden = self.encoder(input_ids=ids[None].to(self.device), attention_mask=masks,
                                  position_ids=positions[None].to(self.device)).last_hidden_state[0].float()
            in_option = groups >= 0
            sums = self.torch.zeros(len(options), hidden.shape[-1]).index_add_(0, groups[in_option], hidden[in_option])
            pooled = sums / self.torch.bincount(groups[in_option], minlength=len(options))[:, None]
            return self.head(pooled).squeeze(-1).cpu()

    def packed_row(self, state, question, options):
        """(token ids, groups, positions) as LongTensors: state, question, then each option restarting its positions
        right after the question."""
        tok = self.tokenizer
        state_ids = tok(state)["input_ids"]
        question_ids = tok(f"Question: {question}\nOption:", add_special_tokens=False)["input_ids"]
        start = len(state_ids) + len(question_ids)
        ids, groups = state_ids + question_ids, [STATE_GROUP] * len(state_ids) + [QUESTION_GROUP] * len(question_ids)
        positions = list(range(start))
        for k, option in enumerate(options):
            option_ids = tok(" " + option, add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
            ids, groups = ids + option_ids, groups + [k] * len(option_ids)
            positions += range(start, start + len(option_ids))
        return tuple(self.torch.tensor(x, dtype=self.torch.long) for x in (ids, groups, positions))
