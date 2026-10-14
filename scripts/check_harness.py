#!/usr/bin/env python3
"""Regression gate: the packaged router must reproduce the intent-eval harness's reference predictions exactly.

Builds each device from the harness's own files (manifest, examples, synthetic + dialog negatives, tools, prompts)
and replays LLM answers from its cache/llm_cache.jsonl. No model calls: a cache miss fails the check, because the
references were measured on cached answers. References: preds/v7__<device>__<route>__<SPLIT>.jsonl: v5 plus the
word-boundary matcher, the catch-all yield at 0.8 and float32 classifier features, written by intent-eval
refs_v6.py (its v7 mode) from the experiment's own implementation of the rule.

Usage (from the switchboard repo, in an environment with the package installed):
    python scripts/check_harness.py ~/coding/OscillateLabs/intent-eval [--quick]
"""
import json
import os
import sys

from ovos_switchboard.adapters.openai_llm import LLMClient, LLMEndpoint
from ovos_switchboard.core.device import Device, read_jsonl
from ovos_switchboard.core.options import ClassifierOptions
from ovos_switchboard.core.ports import LLMUnavailable
from ovos_switchboard.core.router import Router, RouterConfig
from ovos_switchboard.core.schema import Schema


# Must match the harness's llm_fallback.py: the model name is part of every cached request body.
HARNESS_MODEL = "Qwen/Qwen3-14B-AWQ"
DEVICES = {
    "mac": {"manifest": "devices/desktop/manifest.json", "examples": "devices/desktop/examples.jsonl",
            "synthetic": ["skills/synthetic_examples_slotted.jsonl", "devices/desktop/dialog_negatives.jsonl"],
            "tools": "devices/desktop/tools.json"},
    "hub": {"manifest": "devices/speaker/manifest.json", "examples": "devices/speaker/examples.jsonl",
            "synthetic": ["skills/synthetic_examples_slotted.jsonl", "devices/speaker/dialog_negatives.jsonl"],
            "tools": "devices/speaker/tools.json", "shared_examples": ["devices/desktop/examples.jsonl"],
            "shared_manifests": ["devices/desktop/manifest.json"]},
}


class CacheOnlyClient(LLMClient):
    """Answers only from the harness's LLM cache."""
    def _post(self, body):
        raise LLMUnavailable("not in the harness cache; references were measured on cached answers")

    def _store(self, key, entry):
        raise AssertionError("cache-only client never stores")


def build_router(harness, device, route):
    """The router for one harness device, built from the harness's own files."""
    spec = DEVICES[device]

    def path(name):
        return os.path.join(harness, name)

    def prompt(name):
        with open(path(f"prompts/{name}"), encoding="utf-8") as f:
            return f.read().strip()

    dev = Device.from_files(path(spec["manifest"]), path(spec["examples"]),
                            shared_examples=[path(p) for p in spec.get("shared_examples", [])],
                            shared_manifests=[path(p) for p in spec.get("shared_manifests", [])])
    synthetic = [r for p in spec["synthetic"] for r in read_jsonl(path(p))]
    with open(path(spec["tools"]), encoding="utf-8") as f:
        tools_doc = json.load(f)
    llm = CacheOnlyClient(LLMEndpoint(url="http://cache-only.invalid", model=HARNESS_MODEL),
                          system_prompt=prompt("fallback_strict.txt"), gate_prompt=prompt("addressee_gate.txt"),
                          cache_path=path("cache/llm_cache.jsonl"))
    return Router(dev, Schema.load(path("schema/intents.yaml")), tools_doc, synthetic=synthetic, llm=llm,
                  config=RouterConfig(classifier=ClassifierOptions(route_decline_class=route)))


def reference_intents(path):
    """{id: intent} from a harness predictions file."""
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r.get("intent") for r in map(json.loads, f)}


def check(cp, router, reference_path, split):
    """Replay one split through the router; True if every decision matches the reference."""
    expected = reference_intents(reference_path)
    gold, _, _ = cp.setup(split)
    _, _, _, ctx = cp.load_inputs(split)
    bad = []
    for rid, r in gold.items():
        c = ctx.get(rid, cp.COLD)
        got = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]}).get("intent")
        if got != expected[rid]:
            bad.append((rid, r["utterance"], expected[rid], got))
    print(f"{os.path.basename(reference_path)}: {len(gold) - len(bad)}/{len(gold)} identical")
    for b in bad[:8]:
        print("   MISMATCH", b)
    return not bad


def main(argv):
    """Check every device, route policy and split; exit 1 on any mismatch."""
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    harness = os.path.abspath(os.path.expanduser(args[0]))
    sys.path.insert(0, harness)
    os.chdir(harness)  # the harness writes and reads relative to its root
    from switchboard_eval.harness import context_policy as cp  # pylint: disable=import-error  # the harness checkout
    splits = ["dev"] if "--quick" in argv else ["dev", "test"]
    ok = True
    for device in DEVICES:
        for route in ("always", "active"):
            router = build_router(harness, device, route)
            results = [check(cp, router, os.path.join(harness, f"preds/v7__{device}__{route}__{s.upper()}.jsonl"), s)
                       for s in splits]  # every split reports
            ok = ok and all(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
