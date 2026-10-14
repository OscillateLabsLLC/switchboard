"""LLM tool definitions derived ONLY from the device's intent manifest: no skill changes, no hand-written
descriptions. intents.yaml is deliberately not read here; tools stand on what skills register.

Per registration:
  name         short, API-safe (^[a-zA-Z0-9_-]{1,64}$); `labels` maps it back to skill_id:intent_name
  description  humanized intent name + skill + up to MAX_EXAMPLES phrasings (templates or keyword combinations)
  parameters   one string per {slot}; required iff every template guarantees the slot;
               slot_blacklist becomes "never one of: ..."
Plus META_TOOLS for paths the manifest can't express (knowledge, stop, ignore, unsupported).
"""
import logging
import re

from ovos_switchboard.core.templates import SLOT, expand


LOG = logging.getLogger(__name__)

MAX_EXAMPLES = 4
MAX_EXPANSIONS_PER_LINE = 50
SKILL_AFFIXES = re.compile(r"^(ovos-skill-|skill-ovos-|skill-)|\.(openvoiceos|oscillatelabsllc|neongeckocom)$")
META_PREFIX = "meta:"

# Descriptions are part of the measured prompt: changing their text changes the LLM's routing.
META_TOOLS = {
    "ask_knowledge": (("A question or conversational request directed at the assistant that no other "
                       "tool handles: facts, explanations, advice, chit-chat, quotes."),
                      {"question": "the request, as spoken"}, ["question"]),
    "stop": (("Stop whatever the assistant is currently doing: speaking, playing media, an "
              "active timer or alarm sound."), {}, []),
    "unsupported": (("A command clearly directed at the assistant that none of the other tools can "
                     "perform. The assistant will say it can't do that."),
                    {"request": "the request, as spoken"}, ["request"]),
    "ignore": (("The speech was NOT directed at the assistant: background conversation, TV or "
                "radio, the assistant hearing its own voice, or noise. The assistant stays silent."),
               {}, []),
}


def short_skill(skill_id):
    """A skill id without vendor affixes, safe for a tool name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", SKILL_AFFIXES.sub("", skill_id)).strip("_")


def humanize(intent_name):
    """'what_time_is_it' / 'HandleCurrentWeather' -> 'what time is it' / 'current weather'."""
    words = re.sub(r"[._-]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", intent_name)).lower().split()
    if len(words) > 1 and words[-1] == "intent":
        words = words[:-1]
    if len(words) > 1 and words[0] == "handle":  # Neon Adapt handler naming
        words = words[1:]
    return " ".join(words)


def guaranteed_slots(template):
    """Slots present in EVERY expansion: [optional] groups add none, (a|b) adds the intersection."""
    out, i = set(), 0
    while i < len(template):
        ch = template[i]
        if ch in "([":
            j = _matching_close(template, i)
            if ch == "(":
                out |= set.intersection(*(guaranteed_slots(b) for b in _split_branches(template[i + 1:j])))
            i = j + 1
            continue
        m = SLOT.match(template, i)
        if m:
            out.add(m.group(1))
            i = m.end()
            continue
        i += 1
    return out


def _matching_close(t, start):
    depth = 0
    for j in range(start, len(t)):
        depth += t[j] in "(["
        depth -= t[j] in ")]"
        if depth == 0:
            return j
    raise ValueError(f"unbalanced template: {t!r}")


def _split_branches(inner):
    parts, depth, cur = [], 0, ""
    for ch in inner:
        depth += ch in "(["
        depth -= ch in ")]"
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def example_phrasings(samples):
    """Up to MAX_EXAMPLES phrasings, slots shown as <slot>: start from the shortest expansion, then greedily add the
    candidate sharing the fewest words with those already picked (near-duplicates waste the description)."""
    candidates = {}
    for line in samples:
        for e in expand(line)[:MAX_EXPANSIONS_PER_LINE]:
            e = SLOT.sub(lambda m: f"<{m.group(1)}>", re.sub(r"\s+", " ", e).strip())
            if e:
                candidates.setdefault(e.lower(), e)
    pool = sorted(candidates.values(), key=lambda e: (len(e), e))
    picked = pool[:1]
    while pool and len(picked) < min(MAX_EXAMPLES, len(pool)):
        rest = [e for e in pool if e not in picked]
        picked.append(min(rest, key=lambda e: (max(_overlap(e, p) for p in picked), len(e))))
    return picked


def _overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / len(wa | wb)


def _function(skill_id, intent_name, description, props, required):
    return {"type": "function", "function": {
        "name": f"{short_skill(skill_id)}__{re.sub(r'[^a-zA-Z0-9]+', '_', intent_name)}"[:64],
        "description": f"{humanize(intent_name).capitalize()} ({short_skill(skill_id).replace('_', ' ')} skill). "
                       + description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


def template_tool(skill_id, intent_name, definition):
    """Function tool for a template intent: its slots as parameters, example phrasings."""
    samples = definition.get("samples") or []
    slots = sorted({s for line in samples for s in SLOT.findall(line)})
    required = sorted(set.intersection(*(guaranteed_slots(line) for line in samples))) if samples else []
    blacklist = definition.get("slot_blacklist") or {}
    props = {}
    for s in slots:
        desc = s.replace("_", " ")
        if blacklist.get(s):
            desc += ". Never one of: " + ", ".join(blacklist[s])
        props[s] = {"type": "string", "description": desc}
    examples = "; ".join(f'"{e}"' for e in example_phrasings(samples))
    return _function(skill_id, intent_name, f"Example requests: {examples}", props, required)


def adapt_tool(skill_id, intent_name, definition):
    """Keyword intents declare keywords, not phrases or slots: describe them by keyword combinations."""
    examples = "; ".join(f'"{e}"' for e in (definition.get("keyword_examples") or [])[:MAX_EXAMPLES])
    return _function(skill_id, intent_name, f"Triggered by keywords such as: {examples}", {}, [])


def meta_tool(name, desc, params, required):
    """Function tool for a pipeline action (ignore, ask_knowledge, ...)."""
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "required": required,
                       "properties": {k: {"type": "string", "description": v} for k, v in params.items()}}}}


def build_tools(definitions, lang="en-us"):
    """{"tools": [...], "labels": {tool_name: "skill_id:intent_name" | "meta:<name>"}}"""
    tools, labels = [], {}
    for skill_id, defs in definitions.items():
        for d in defs if isinstance(defs, list) else []:
            if d["lang"].lower() != lang or d["method"] not in ("template", "adapt"):
                continue
            make = adapt_tool if d["method"] == "adapt" else template_tool
            t = make(skill_id, d["intent_name"], d["definition"])
            name = t["function"]["name"]
            if name in labels:
                if labels[name] != f"{skill_id}:{d['intent_name']}":  # same intent in another lang/method is fine
                    LOG.warning("tool name collision: %s for %s and %s:%s; keeping the first",
                                name, labels[name], skill_id, d["intent_name"])
                continue
            labels[name] = f"{skill_id}:{d['intent_name']}"
            tools.append(t)
    for name, spec in META_TOOLS.items():
        tools.append(meta_tool(name, *spec))
        labels[name] = META_PREFIX + name
    return {"tools": tools, "labels": labels}


# The decision model's options, in plain words. Measured wording (intent-eval exp_decider.py, plain framing): the
# LLM's tool descriptions gave a decision model nothing to work with.
DECIDER_QUESTION = "What does the speaker want the voice assistant to do?"
DECIDER_NONE = "ignore"
DECIDER_META = {"ask_knowledge": "a general question or chit-chat for the assistant",
                "stop": "stop what the assistant is doing",
                DECIDER_NONE: "none of these: not a request for the assistant"}
QUOTED = re.compile(r'"([^"]+)"')
ANGLE_SLOT = re.compile(r"<([^>]+)>")


def plain_option(tool):
    """A tool as a decision-model option: its description's first sentence and up to 2 examples, slots named.
    None for meta tools a decision model isn't offered (unsupported; ignore is offered as DECIDER_META's NONE)."""
    name, desc = tool["function"]["name"], tool["function"]["description"]
    if name in DECIDER_META:
        return None if name == DECIDER_NONE else DECIDER_META[name]
    if name not in META_TOOLS:
        first = desc.split(". ")[0].strip()
        examples = [ANGLE_SLOT.sub(r"\1", e) for e in QUOTED.findall(desc)[:2]]
        return f"{first}: {'; '.join(examples)}" if examples else first
    return None


def decider_options(tools):
    """{tool name: plain option} for the offered tools, plus "none of these" last."""
    options = {t["function"]["name"]: plain_option(t) for t in tools}
    options = {k: v for k, v in options.items() if v}
    options[DECIDER_NONE] = DECIDER_META[DECIDER_NONE]
    return options
