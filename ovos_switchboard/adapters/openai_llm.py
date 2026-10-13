"""LLM adapter: the LLMPort over any OpenAI-compatible chat completions endpoint (vLLM, llama.cpp, OpenAI, ...).

Two calls per utterance, both temperature 0:
  route        tool_choice=required over the shortlisted tools, so the model must pick one (`ignore` included)
  p_addressed  P(yes) to "is this directed at the assistant?", from first-token logprobs (the addressee gate)
Responses are cached by a hash of the request body: identical requests are answered from the cache, which makes
repeats deterministic and lets the harness re-score without a model. Any transport or protocol failure raises
LLMUnavailable; the caller decides what that means (the plugin declines).
"""
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass

from ovos_switchboard.core.ports import LLMPort, LLMUnavailable


GATE_TOP_LOGPROBS = 10  # the gate reads yes/no mass from this many first-token alternatives


@dataclass(frozen=True)
class LLMEndpoint:
    """Where and how to call the model. Field names are the LLM step's config keys."""
    url: str
    model: str = ""
    api_key: str | None = None
    timeout_s: float = 4.0          # per request
    route_max_tokens: int = 200     # room for one tool call with its arguments
    cache_size: int = 1024          # in-memory response cache entries (a cache file is unbounded)


class LLMClient(LLMPort):
    """LLMPort over an OpenAI-compatible endpoint, with a response cache keyed by request body."""
    def __init__(self, endpoint, system_prompt, gate_prompt, cache_path=None):
        self.endpoint = endpoint
        self.system_prompt, self.gate_prompt = system_prompt, gate_prompt
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        if cache_path:
            try:
                with open(cache_path, encoding="utf-8") as f:
                    for line in f:
                        e = json.loads(line)
                        self.cache[e["key"]] = e
            except FileNotFoundError:
                pass

    def route(self, utterance, tools):
        """{"tool": name, "args": {...}, "ms": float}"""
        body = {"model": self.endpoint.model, "temperature": 0, "max_tokens": self.endpoint.route_max_tokens,
                "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False},
                "tools": tools,
                "messages": [{"role": "system", "content": self.system_prompt},
                             {"role": "user", "content": utterance}]}
        return self._cached(body, parse_tool_call, {"n_tools": len(tools)})

    def p_addressed(self, utterance):
        """{"p_yes": float, "ms": float}"""
        body = {"model": self.endpoint.model, "temperature": 0, "max_tokens": 1, "logprobs": True,
                "top_logprobs": GATE_TOP_LOGPROBS, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": self.gate_prompt},
                             {"role": "user", "content": utterance}]}
        return self._cached(body, parse_yes_no)

    def _cached(self, body, parse, extra=None):
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return {**self.cache[key], "cached": True}
        t0 = time.perf_counter()
        r = self._post(body)
        try:
            entry = {"key": key, **parse(r), "ms": round((time.perf_counter() - t0) * 1000, 1),
                     "prompt_tokens": r.get("usage", {}).get("prompt_tokens"), **(extra or {})}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMUnavailable(f"unusable LLM response: {e}") from e
        self._store(key, entry)
        return {**entry, "cached": False}

    def _post(self, body):
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        req = urllib.request.Request(self.endpoint.url, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.endpoint.timeout_s) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise LLMUnavailable(f"{self.endpoint.url}: {e}") from e

    def _store(self, key, entry):
        with self.lock:
            self.cache[key] = entry
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            else:
                while len(self.cache) > self.endpoint.cache_size:
                    self.cache.popitem(last=False)


def parse_tool_call(r):
    """The first tool call in a chat completion as {"tool", "args"}."""
    calls = r["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        raise ValueError(f"tool_choice=required returned no tool call: {r['choices'][0]['message']}")
    fn = calls[0]["function"]
    return {"tool": fn["name"], "args": json.loads(fn["arguments"] or "{}")}


def parse_yes_no(r):
    """Renormalize over yes/no variants in the top logprobs; fail if neither appears."""
    mass = {"yes": 0.0, "no": 0.0}
    for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        word = t["token"].strip(" _").lower()
        if word in mass:
            mass[word] += math.exp(t["logprob"])
    if not sum(mass.values()):
        raise ValueError(f"gate answer has no yes/no token: {r['choices'][0]}")
    return {"p_yes": mass["yes"] / (mass["yes"] + mass["no"])}
