"""Model cache adapter: trained models on local disk, keyed by a hash of everything they were trained from.

A restart with unchanged skills loads the classifier instead of retraining it (about 20 s on a Mac). Models are
stored as plain data, never pickled: numpy arrays plus a JSON string in one .npz file, read with
allow_pickle=False, so a tampered or corrupt file can fail to load but can't run code in ovos-core. Writes are
atomic, the directory is owner-only, and only the newest `max_entries` per kind are kept. A file that fails to load
is retrained and replaced.
"""
import glob
import hashlib
import json
import logging
import os
import tempfile

import numpy as np


LOG = logging.getLogger(__name__)
META_KEY = "__meta__"


def default_cache_dir():
    """switchboard under $XDG_CACHE_HOME (default ~/.cache)."""
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "switchboard")


class ModelCache:
    """Get a trained model by its inputs, training it only when nothing cached matches."""

    def __init__(self, path=None, max_entries=2, enabled=True):
        self.path, self.max_entries, self.enabled = path or default_cache_dir(), max_entries, enabled

    def get_or_build(self, kind, key_parts, build, codec):
        """(model, cache_hit). key_parts must be JSON-serializable and cover every input that changes the model.
        codec.state(model) -> (arrays, meta) and codec.from_state(arrays, meta) -> model convert to plain data."""
        if not self.enabled:
            return build(), False
        digest = hashlib.sha256(json.dumps(key_parts, sort_keys=True, default=str).encode()).hexdigest()[:32]
        path = os.path.join(self.path, f"{kind}-{digest}.npz")
        model = self._load(path, codec)
        if model is not None:
            return model, True
        model = build()
        self._store(kind, path, codec.state(model))
        return model, False

    @staticmethod
    def _load(path, codec):
        try:
            with np.load(path, allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in npz.files if k != META_KEY}
                meta = json.loads(str(npz[META_KEY]))
            model = codec.from_state(arrays, meta)
        except FileNotFoundError:
            return None
        except Exception:
            LOG.warning("unreadable cached model %s; retraining", path)
            return None
        os.utime(path)  # recently used entries survive pruning
        return model

    def _store(self, kind, path, state):
        arrays, meta = state
        tmp = None
        try:
            os.makedirs(self.path, mode=0o700, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path, suffix=".tmp", delete=False) as f:
                tmp = f.name
                np.savez(f, **{META_KEY: np.array(json.dumps(meta))}, **arrays)
            os.replace(tmp, path)
        except OSError as e:
            LOG.warning("could not cache the %s model: %s", kind, e)
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            return
        entries = sorted(glob.glob(os.path.join(self.path, f"{kind}-*.npz")), key=os.path.getmtime, reverse=True)
        for stale in entries[self.max_entries:]:
            os.remove(stale)
