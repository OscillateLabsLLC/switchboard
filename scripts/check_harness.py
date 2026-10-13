#!/usr/bin/env python3
"""Regression gate: the packaged router must reproduce the intent-eval harness's reference predictions exactly.

Builds each device from the harness's own files (manifest, examples, synthetic + dialog negatives, tools, prompts)
and replays LLM answers from its llm_cache.jsonl. No model calls: a cache miss fails the check, because the
references were measured on cached answers. References: preds/v5__<device>__<route>__<SPLIT>.jsonl.

Usage (from the switchboard repo, in an environment with the package installed):
    python scripts/check_harness.py ~/coding/OscillateLabs/intent-eval [--quick]
"""
import json
import os
import sys

from ovos_switchboard.device import Device
from ovos_switchboard.llm import LLMClient, LLMUnavailable
from ovos_switchboard.router import Router, RouterConfig
from ovos_switchboard.schema import Schema

# Must match the harness's llm_fallback.py: the model name is part of every cached request body.
HARNESS_MODEL = "Qwen/Qwen3-14B-AWQ"
DEVICES = {
    "mac": {"manifest": "introspection.json", "examples": "examples_live.jsonl",
            "synthetic": ["synthetic_examples_slotted.jsonl", "dialog_negatives.jsonl"], "tools": "tools.json"},
    "hub": {"manifest": "introspection_hub.json", "examples": "examples_hub.jsonl",
            "synthetic": ["synthetic_examples_slotted.jsonl", "dialog_negatives_hub.jsonl"], "tools": "tools_hub.json",
            "shared_examples": ["examples_live.jsonl"], "shared_manifests": ["introspection.json"]},
}


class CacheOnlyClient(LLMClient):
    def _post(self, body):
        raise LLMUnavailable("not in the harness cache; references were measured on cached answers")

    def _store(self, key, entry):
        raise AssertionError("cache-only client never stores")


def build_router(harness, device, route):
    spec = DEVICES[device]

    def path(name):
        return os.path.join(harness, name)

    def prompt(name):
        with open(path(f"prompts/{name}")) as f:
            return f.read().strip()

    dev = Device.from_files(path(spec["manifest"]), path(spec["examples"]), [path(p) for p in spec["synthetic"]],
                            shared_examples=[path(p) for p in spec.get("shared_examples", [])],
                            shared_manifests=[path(p) for p in spec.get("shared_manifests", [])])
    with open(path(spec["tools"])) as f:
        tools_doc = json.load(f)
    llm = CacheOnlyClient(url="http://cache-only.invalid", model=HARNESS_MODEL,
                          system_prompt=prompt("fallback_strict.txt"), gate_prompt=prompt("addressee_gate.txt"),
                          cache_path=path("llm_cache.jsonl"))
    return Router(dev, Schema.load(path("intents.yaml")), tools_doc, llm, RouterConfig(route_decline_class=route))


def check(harness, cp, device, route, split, router):
    with open(os.path.join(harness, f"preds/v5__{device}__{route}__{split.upper()}.jsonl")) as f:
        ref = {r["id"]: r for r in map(json.loads, f)}
    gold, _, _ = cp.setup(split)
    _, _, _, ctx = cp.load_inputs(split)
    bad = []
    for rid, r in gold.items():
        c = ctx.get(rid, cp.COLD)
        got = router.route(r["utterance"], {"gap_s": c["gap_s"], "echo_run": c["echo_run"]})
        if got.get("intent") != ref[rid].get("intent"):
            bad.append((rid, r["utterance"], ref[rid].get("intent"), got.get("intent")))
    print(f"{device} {route} {split}: {len(gold) - len(bad)}/{len(gold)} identical")
    for b in bad[:8]:
        print("   MISMATCH", b)
    return not bad


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    harness = os.path.abspath(os.path.expanduser(args[0]))
    sys.path.insert(0, harness)
    os.chdir(harness)  # context_policy reads its inputs relative to the harness
    import context_policy as cp
    splits = ["dev"] if "--quick" in argv else ["dev", "test"]
    ok = True
    for device in DEVICES:
        for route in ("always", "active"):
            router = build_router(harness, device, route)
            results = [check(harness, cp, device, route, s, router) for s in splits]  # every split reports
            ok = ok and all(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
