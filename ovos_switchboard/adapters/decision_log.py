"""Decision log adapter: JSONL, one record per stage verdict plus one per OVOS dispatch outcome, so shadow mode can be
compared offline with what the device actually did. It holds household speech: it stays local and rotates by size.
"""
import json
import os
import threading
import time


def default_log_path():
    state = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(state, "switchboard", "decisions.jsonl")


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
