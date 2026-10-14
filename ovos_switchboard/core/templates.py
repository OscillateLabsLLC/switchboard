"""Padatious-style template helpers: expansion, matching regexes, literal-word counts.

Template syntax as skills write it in .intent files: (a|b) alternation, [optional] groups, {slot} wildcards.
"""
import itertools
import re


MAX_PER_LINE = 200
LITERAL_EXPANSION_LIMIT = 50  # expansions examined when counting a template's literal words
SLOT = re.compile(r"\{+([a-z_]+)\}+")
SLOT_TOKEN = re.compile(r"\{+[a-z_]+\}+")
# A template space: whitespace, or nothing where an omitted [optional] group leaves a word edge. Never zero
# characters inside a word, so "play {query}" doesn't match "playing by".
WORD_GAP = r"(?:\s+|\b)"


def split_top(s, sep="|"):
    """Split on `sep` outside any (...) or [...] group."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def expand(t):
    """Expand (a|b|), [optional] groups. {slots} are left in place."""
    for i, ch in enumerate(t):
        if ch in "([":
            close = ")" if ch == "(" else "]"
            depth = 0
            for j in range(i, len(t)):
                if t[j] in "([":
                    depth += 1
                elif t[j] in ")]":
                    depth -= 1
                    if depth == 0:
                        break
            inner = t[i + 1:j]
            opts = split_top(inner) if ch == "(" else [inner, ""]
            if close == "]" and "|" in inner:
                opts = split_top(inner) + [""]
            out = []
            for o in opts:
                out.extend(expand(t[:i] + o + t[j + 1:]))
            return out
    return [t]


def clean(s):
    """Collapse whitespace."""
    return re.sub(r"\s+", " ", s).strip()


def expansions(line, limit=MAX_PER_LINE):
    """Cleaned, non-empty expansions of a template line, at most `limit`."""
    return [c for c in (clean(e) for e in itertools.islice(expand(line), limit)) if c]


def template_regex(line):
    """Template -> regex matching any expansion, slots as wildcards."""
    return re.compile(r"^\s*" + _pattern(line, named=False) + r"\s*[?.!]*$")


def named_template_regex(line):
    """Pattern body with slots as named groups (repeats get __N suffixes), unanchored."""
    return _pattern(line, named=True)


def _pattern(line, named):
    pat, i, seen = "", 0, {}
    line = line.lower().replace("{{", "{").replace("}}", "}")
    while i < len(line):
        ch = line[i]
        if ch == "{":
            j = line.index("}", i)
            if named:
                name = line[i + 1:j]
                seen[name] = seen.get(name, 0) + 1
                pat += f"(?P<{name}__{seen[name]}>.+?)"
            else:
                pat += r"(?:.+?)"
            i = j + 1
            continue
        pat += {"(": "(?:", ")": ")", "[": "(?:", "]": ")?", "|": "|", " ": WORD_GAP}.get(ch, re.escape(ch))
        i += 1
    return pat


def literal_words(line, limit=None):
    """Fewest literal (non-slot) words in any expansion. 'play {query}' -> 1."""
    if not line:
        return 0
    found = expand(line) if limit is None else expand(line)[:limit]
    return min(len(SLOT_TOKEN.sub(" ", e).split()) for e in found)


def is_catch_all(line):
    """A slot behind fewer than two literal words ('play {query}', 'call {contact}'): it claims almost any speech
    that starts with its one word."""
    return "{" in line and literal_words(line) < 2


def compiles(line):
    """Whether the template's named-group regex compiles (slot extraction). Typed slots ({number:priority}) fail
    here, since ':' can't be in a group name, but still match: see well_formed."""
    try:
        re.compile(named_template_regex(line))
        return True
    except (ValueError, re.error):
        return False


def well_formed(line):
    """Whether the template's matching regex compiles: false only for broken syntax such as an unclosed group."""
    try:
        template_regex(line)
        return True
    except (ValueError, re.error):
        return False
