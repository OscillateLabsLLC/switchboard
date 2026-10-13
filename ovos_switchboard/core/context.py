"""Live addressee context from the assistant's own speech, the features the harness measured from device logs:
  gap_s     seconds since the assistant most recently started speaking (None if it hasn't, as far as we know)
  echo_run  longest run of contiguous words the utterance shares with anything the assistant said in the last
            ECHO_WINDOW_S (requests reuse the assistant's WORDS; echo copies its PHRASES)
"""
import re
import threading
import time
from collections import defaultdict, deque
from difflib import SequenceMatcher

ECHO_WINDOW_S = 60
HISTORY_S = 600
MAX_LINES_PER_SESSION = 64


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", ""))).strip()


def longest_run(a, b):
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


class SpeechContext:
    """Thread-safe record of what the assistant said, per session."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.spoken = defaultdict(lambda: deque(maxlen=MAX_LINES_PER_SESSION))  # session -> (ts, words)

    def heard_assistant(self, text, session_id="default", ts=None):
        if not text:
            return
        ts = self.clock() if ts is None else ts
        with self.lock:
            lines = self.spoken[session_id]
            lines.append((ts, norm(text).split()))
            while lines and ts - lines[0][0] > HISTORY_S:
                lines.popleft()

    def features(self, utterance, session_id="default", now=None):
        now = self.clock() if now is None else now
        with self.lock:
            lines = [(t, w) for t, w in self.spoken.get(session_id, ()) if t <= now]
        if not lines:
            return {"gap_s": None, "echo_run": 0}
        words = norm(utterance).split()
        recent = [w for t, w in lines if now - t <= ECHO_WINDOW_S]
        return {"gap_s": round(now - lines[-1][0], 1),
                "echo_run": max((longest_run(words, r) for r in recent), default=0)}
