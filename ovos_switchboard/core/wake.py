"""Leading wake phrases in transcripts ("Hey Mycroft, where were you born").

Speech-to-text sometimes keeps the wake word, and an anchored template can't match past it. The exact and prefix
stages strip it before matching. Measured: Hub 0.3% of transcripts, Mac 1.5%.
"""
import re


def wake_phrase_pattern(phrases):
    """Regex for one leading wake phrase from names like "hey_mycroft" or "hey mycroft", or None without any."""
    alternatives = {r"[\s,.!?]*".join(map(re.escape, words))
                    for words in (re.findall(r"[a-z0-9]+", p.lower()) for p in phrases) if words}
    if not alternatives:
        return None
    ordered = sorted(alternatives, key=len, reverse=True)
    return re.compile(r"^\W*(?:" + "|".join(ordered) + r")\b[\s,.!?]*", re.IGNORECASE)


def strip_wake_phrase(utterance, pattern):
    """`utterance` without a leading wake phrase. A bare wake phrase is returned unchanged."""
    if pattern is None:
        return utterance
    rest = pattern.sub("", utterance, count=1)
    return rest if rest.strip() else utterance
