"""Per-utterance decision cache (the three pipeline tiers share one routing computation) and the decision log.

The log is JSONL, one record per routed utterance plus one per OVOS dispatch outcome, so shadow mode can be compared
offline with what the device actually did. It holds household speech: it stays local and rotates by size.
"""
import json
import os
import threading
import time
from collections import OrderedDict

CACHE_SIZE = 64
CACHE_TTL_S = 60


def default_log_path():
    state = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(state, "switchboard", "decisions.jsonl")


class DecisionCache:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.items = OrderedDict()

    def get(self, key):
        with self.lock:
            hit = self.items.get(key)
            if hit and self.clock() - hit[0] <= CACHE_TTL_S:
                return hit[1]
            return None

    def put(self, key, value):
        with self.lock:
            self.items[key] = (self.clock(), value)
            self.items.move_to_end(key)
            while len(self.items) > CACHE_SIZE:
                self.items.popitem(last=False)


class DecisionLog:
    def __init__(self, path=None, max_bytes=5_000_000, enabled=True):
        self.path, self.max_bytes, self.enabled = path or default_log_path(), max_bytes, enabled
        self.lock = threading.Lock()
        if enabled:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, record):
        if not self.enabled:
            return
        line = json.dumps({"ts": round(time.time(), 3), **record}, default=str) + "\n"
        with self.lock:
            if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
                os.replace(self.path, f"{self.path}.1")
            with open(self.path, "a") as f:
                f.write(line)
