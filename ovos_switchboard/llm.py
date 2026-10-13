"""Stage 5: an OpenAI-compatible chat endpoint routes what the earlier stages could not.

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
from importlib import resources

GATE_TOP_LOGPROBS = 10
ROUTE_MAX_TOKENS = 200
MEMORY_CACHE_SIZE = 1024


class LLMUnavailable(RuntimeError):
    """The endpoint could not be reached or answered something unusable."""


def bundled_prompt(name):
    return resources.files("ovos_switchboard.data.prompts").joinpath(name).read_text().strip()


class LLMClient:
    def __init__(self, url, model, system_prompt, gate_prompt, timeout_s=10.0, api_key=None, cache_path=None):
        self.url, self.model, self.timeout_s, self.api_key = url, model, timeout_s, api_key
        self.system_prompt, self.gate_prompt = system_prompt, gate_prompt
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        if cache_path:
            try:
                with open(cache_path) as f:
                    for line in f:
                        e = json.loads(line)
                        self.cache[e["key"]] = e
            except FileNotFoundError:
                pass

    def route(self, utterance, tools):
        """{"tool": name, "args": {...}, "ms": float}"""
        body = {"model": self.model, "temperature": 0, "max_tokens": ROUTE_MAX_TOKENS,
                "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False},
                "tools": tools,
                "messages": [{"role": "system", "content": self.system_prompt},
                             {"role": "user", "content": utterance}]}
        return self._cached(body, parse_tool_call, {"n_tools": len(tools)})

    def p_addressed(self, utterance):
        """{"p_yes": float, "ms": float}"""
        body = {"model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True,
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
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise LLMUnavailable(f"{self.url}: {e}") from e

    def _store(self, key, entry):
        with self.lock:
            self.cache[key] = entry
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            else:
                while len(self.cache) > MEMORY_CACHE_SIZE:
                    self.cache.popitem(last=False)


def parse_tool_call(r):
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
