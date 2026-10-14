"""Train the classifier in a child process, so its scratch memory never stays in ovos-core.

The logistic-regression fit (L-BFGS) needs gigabytes of temporary memory at a few hundred intents (3.5 GB at 465,
intent-eval exp_memory_phases.py), and the allocator keeps that high-water mark after the fit, while the trained
model is ~200 MB. A spawned child (not forked: ovos-core runs threads) trains, writes the model as plain arrays in
the model cache's no-pickle .npz format, and exits; the parent only loads the arrays. Same code, same data, so the
model is the one in-process training would give.
"""
import json
import multiprocessing
import os
import tempfile

import numpy as np

from ovos_switchboard.adapters.model_cache import META_KEY
from ovos_switchboard.core.classifier import Classifier


class TrainingFailed(RuntimeError):
    """The training child exited without writing a model."""


def train_isolated(rows):
    """Classifier.train(rows), run in a spawned child process."""
    with tempfile.TemporaryDirectory(prefix="switchboard-train-") as tmp:
        path = os.path.join(tmp, "classifier.npz")
        child = multiprocessing.get_context("spawn").Process(target=_train_to_file, args=(rows, path),
                                                             name="switchboard-train", daemon=True)
        child.start()
        child.join()
        if child.exitcode != 0 or not os.path.exists(path):
            raise TrainingFailed(f"classifier training child exited with {child.exitcode}")
        with np.load(path, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files if k != META_KEY}
            meta = json.loads(str(npz[META_KEY]))
    return Classifier.from_state(arrays, meta)


def _train_to_file(rows, path):
    """Child process: train and write the model's plain-data state."""
    arrays, meta = Classifier.train(rows).state()
    np.savez(path, **{META_KEY: np.array(json.dumps(meta))}, **arrays)
