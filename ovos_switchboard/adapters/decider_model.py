"""DeciderPort over a local typed-decision model (default: tasksource-decider-nano, 150M, Apache-2.0, CPU).

Needs the "decider" extra (torch, transformers). The weights download from the Hugging Face Hub on first use and stay
in its cache; only config, tokenizer and safetensors files are fetched, never code.
"""
import time
from dataclasses import dataclass

from ovos_switchboard.core.ports import DeciderPort


@dataclass(frozen=True)
class DeciderModel:
    """Which model to load. Field names are the decider step's config keys."""
    model: str = "tasksource/tasksource-decider-nano"
    revision: str = "v4"            # the measured weights
    device: str = "cpu"


class TasksourceDecider(DeciderPort):
    """A tasksource-format decider (choice questions over described options)."""

    def __init__(self, spec=DeciderModel()):
        from ovos_switchboard.adapters.vendor.tasksource_decider import Decider  # torch: the "decider" extra
        self.model = Decider.from_pretrained(spec.model, revision=spec.revision, device=spec.device)

    def choose(self, state, question, options):
        t0 = time.perf_counter()
        answer = self.model.answer({"speech": state}, {"q": {"type": "choice", "instructions": question,
                                                             "criteria": options}})["answers"]["q"]
        return {"choice": answer["choice"], "probabilities": answer["probabilities"],
                "ms": round((time.perf_counter() - t0) * 1000, 1)}
