"""DeciderPort over a local typed-decision cross-encoder (default: tasksource-decider-nano, 150M, Apache-2.0, CPU).

The model scores each option as one row, [CLS] state [SEP] "Question: <q>\\nOption:" " <option>" [SEP], with the
three-block attention mask its "shared"/"cached" layout was trained with: the state attends only to itself, the
question to the state and itself, the option to all three. A linear head on the mean of the option block's last
hidden states gives the option's score; probabilities are the softmax over a question's options. (The model's own
inference code packs many options into shared rows to save compute; one short utterance doesn't need that.)

Needs the "decider" extra (torch, transformers). The weights download from the Hugging Face Hub on first use and stay
in its cache; only config, tokenizer and safetensors files are fetched, never code.
"""
import json
import time
from dataclasses import dataclass

from ovos_switchboard.core.ports import DeciderPort


STATE, QUESTION, OPTION = 0, 1, 2
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


def block_mask(kinds, positions, window):
    """Attention allowed (query i -> key j): state sees state; question sees state and question; option sees all
    three; padding (-1) sees only itself. With a window, also |pos_i - pos_j| <= window (local layers)."""
    import torch
    ki, kj = kinds[:, :, None], kinds[:, None, :]
    allow = (kj == STATE) & (ki >= STATE)
    allow |= (kj == QUESTION) & (ki >= QUESTION)
    allow |= (kj == OPTION) & (ki == OPTION)
    allow |= torch.eye(kinds.shape[1], dtype=torch.bool)[None]
    if window is not None:
        allow &= (positions[:, :, None] - positions[:, None, :]).abs() <= window
    return allow[:, None]


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
        """One score per option."""
        rows = [self.row(state, question, o) for o in options]
        ids = self.pad([r[0] for r in rows], self.tokenizer.pad_token_id)
        kinds = self.pad([r[1] for r in rows], -1)
        positions = self.pad([r[2] for r in rows], 0)
        masks = {"full_attention": block_mask(kinds, positions, None)}
        if self.window is not None:
            masks["sliding_attention"] = block_mask(kinds, positions, self.window)
        with self.torch.no_grad():
            hidden = self.encoder(input_ids=ids.to(self.device), attention_mask=masks,
                                  position_ids=positions.to(self.device)).last_hidden_state.float()
            in_option = (kinds == OPTION).to(self.device).unsqueeze(-1)
            pooled = (hidden * in_option).sum(1) / in_option.sum(1)
            return self.head(pooled).squeeze(-1).cpu()

    def row(self, state, question, option):
        """(token ids, block kinds, positions) for one option's row."""
        tok = self.tokenizer
        blocks = [(tok(state)["input_ids"], STATE),
                  (tok(f"Question: {question}\nOption:", add_special_tokens=False)["input_ids"], QUESTION),
                  (tok(" " + option, add_special_tokens=False)["input_ids"] + [tok.sep_token_id], OPTION)]
        ids = [t for block, _ in blocks for t in block]
        kinds = [kind for block, kind in blocks for _ in block]
        return ids, kinds, list(range(len(ids)))

    def pad(self, sequences, fill):
        """A (rows, longest) LongTensor, right-padded with `fill`."""
        width = max(map(len, sequences))
        return self.torch.tensor([s + [fill] * (width - len(s)) for s in sequences], dtype=self.torch.long)
