# Switchboard

Intent routing for OVOS as four pipeline steps. It tries deterministic exact matches first, then a small classifier, then an LLM with tools derived from the skills' own registrations. When speech wasn't meant for the assistant, it stays silent.

Each stage was measured before it was adopted, on the [intent-eval](https://forgejo.graywind.org/graywind/intent-eval) harness, using real household traffic from a Neon Hub and an OVOS Mac. On held-out Hub traffic, Switchboard answered 90.6% of requests correctly versus 42.2% for stock OVOS. On overheard speech, it acted on 7 of 71 utterances where Neon acted on all 71. `scripts/check_harness.py` shows that this package's stages reproduce the harness's measured decisions exactly.

## The steps

Each step is its own OVOS pipeline plugin. You can place, drop or adopt each one on its own.

| pipeline id | decides | outcomes |
|---|---|---|
| `ovos-switchboard-echo-pipeline-plugin` | the utterance repeats 4+ contiguous words the assistant said in the last 60 s | decline |
| `ovos-switchboard-exact-pipeline-plugin` | the phrase matches exactly one skill's template | dispatch |
| `ovos-switchboard-classifier-pipeline-plugin` | char n-gram TF-IDF + logistic regression with a `__decline__` class | dispatch if confidence ≥ 0.5; decline or pass for the decline class (policy); pass otherwise |
| `ovos-switchboard-llm-pipeline-plugin` | an LLM picks one of the shortlisted skill tools; an addressee gate checks P(addressed) ≥ 0.9 | dispatch; decline on `ignore`, a failed gate or an unreachable LLM; pass on `ask_knowledge` / `stop` / `unsupported` |

- **Dispatch** goes to `skill_id:intent_name` with the data that skill reads: template slot values under the template's own names, Adapt vocab matches (`Location`, ...), or LLM tool arguments. Entity resolution and typed parsing stay in the skills.
- **Decline** dispatches to the [Hangup](https://github.com/OscillateLabsLLC/hangup) skill, which speaks nothing and records the utterance and the reason. Hangup is required.
- **Pass** returns no match, so later pipeline steps decide.

Steps never consult each other. Pipeline order already means a step only sees what earlier steps passed on. The classifier owns the decline-class policy (`route_decline_class`), which decides whether a "not for me" verdict declines on the spot or passes on to the LLM:
- `active` (default): passes on only if the assistant spoke within the last 120 s.
- `always`: always passes on.
- `never`: always declines on the spot.

**High-risk intents** (unlock, open covers, shutdown, clear data) are never acted on until a confirm flow exists (`high_risk: "defer"`). Once a step holds an utterance as high-risk, later Switchboard steps can't act on it either.

## Pipeline order (measured configuration)

```json
"intents": {
  "pipeline": [
    "ovos-switchboard-echo-pipeline-plugin",
    "ovos-stop-pipeline-plugin-high",
    "ovos-converse-pipeline-plugin",
    "ovos-switchboard-exact-pipeline-plugin",
    "ovos-ocp-pipeline-plugin-high",
    "ovos-switchboard-classifier-pipeline-plugin",
    "ovos-switchboard-llm-pipeline-plugin",
    "ovos-common-query-pipeline-plugin",
    "ovos-fallback-pipeline-plugin-low"
  ]
}
```

- The **echo step comes first** so the assistant's own voice can't trigger stop or converse.
- The **LLM step runs before common_qa**. If common_qa ran first, it would answer overheard questions before the addressee gate could decline them. Questions the LLM recognizes as addressed to the assistant (`ask_knowledge`) pass through to common_qa.
- **Using a subset is supported.** For example, the exact step alone makes a deterministic replacement for Padatious-high. Results then differ from the measured numbers above. Without an LLM step, set the classifier's `route_decline_class` to `never`, so "not for me" rows decline instead of reaching common_qa.

## Configuration

Shared settings go under `intents.switchboard`. Each step's settings go under its pipeline id, and a step can override the shared `mode`.

```json
"intents": {
  "switchboard": {"mode": "shadow"},
  "ovos-switchboard-llm-pipeline-plugin": {
    "url": "http://llm-host:8080/v1/chat/completions", "model": "Qwen/Qwen3-14B-AWQ", "timeout_s": 4
  }
}
```

| section | key | default | meaning |
|---|---|---|---|
| shared | `mode` | `shadow` | `shadow` logs verdicts and never matches; `active` acts |
| shared | `lang` | `en-us` | language the stages serve; other languages pass |
| shared | `hangup_skill_id` | `ovos-skill-hangup.oscillatelabsllc` | where declines go |
| shared | `high_risk` | `defer` | `defer` or `allow` |
| shared | `retrain_debounce_s` | 5 | quiet time after registration changes before stages rebuild |
| shared | `bus_timeout_s` | 10 | each startup query to ovos-core (intent sync, skills list) |
| shared | `echo_window_s` | 60 | echo compares the utterance against assistant speech this recent |
| shared | `speech_history_s`, `speech_lines_max` | 600, 64 | how much assistant speech is kept per session |
| shared | `record_ttl_s`, `records_max` | 30, 64 | per-utterance bookkeeping (high-risk holds, shadow settlement). Keep `record_ttl_s` above the longest pipeline round |
| shared | `decision_log`, `decision_log_path` | on, `$XDG_STATE_HOME/switchboard/decisions.jsonl` | JSONL of every verdict and OVOS's actual outcome. Holds household speech; stays local |
| echo | `echo_run_min` | 4 | contiguous words shared with recent assistant speech |
| classifier | `threshold` | 0.5 | confidence needed to dispatch |
| classifier | `route_decline_class` | `active` | `never` / `active` / `always` |
| classifier | `active_s` | 120 | "assistant active" window, in seconds |
| classifier | `dialog_negatives` | true | train the decline class on installed skills' `.dialog` lines |
| llm | `url`, `model`, `api_key` | none | OpenAI-compatible endpoint. Without `url` the step never matches |
| llm | `timeout_s` | 4 | per request; the step makes two (route + gate) |
| llm | `route_max_tokens`, `cache_size` | 200, 1024 | room for one tool call; in-memory response cache entries |
| llm | `gate`, `shortlist_k` | 0.9, 12 | addressee threshold; skill tools offered per utterance |

ovos-core abandons any matcher that runs past `intents.<pipeline id>.match_timeout` (default 10 s) and treats it as no match. Keep `2 × timeout_s` under that value, or a slow LLM falls through to common_qa instead of declining.

Every installed step is instantiated by ovos-core even when it isn't in the pipeline. To switch one off entirely, add its id to `intents.blacklisted_pipelines`.

## Shadow mode

In shadow mode every step logs its verdict and latency and returns no match, so the device behaves exactly as before. A later step skips an utterance an earlier step already settled. The LLM step runs off the pipeline thread. OVOS's actual outcome (`ovos.intent.matched` / `unmatched`) goes to the same log for offline comparison. List the Switchboard steps ahead of the stock matchers, or the stock ones will claim utterances first and Switchboard never sees them.

## Architecture

The layout is hexagonal where that pays off:

```
ovos_switchboard/
  core/       domain, no OVOS imports: templates, schema, device data, the four stages (stages.py) returning a
              Verdict (DISPATCH / DECLINE / PASS), the LLM port (ports.py), and router.py, which composes the
              stages in measured order for the harness
  adapters/   driven side: OpenAI-compatible LLM client, JSONL decision log, bus intent registry, dialog files
  ovos/       driving side: the pipeline steps, plus the runtime (composition root) they share per bus
  data/       intents.yaml schema, reviewed synthetic examples, the measured prompts
```

Each step asks the runtime to build only its own stage, so installing the exact step alone trains no classifier. All stages rebuild from the live INTENT-4 registrations in the background when skills register or deregister.

## Development

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test,dev]"
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/pylint ovos_switchboard scripts
```

The regression gate needs the intent-eval checkout and its environment, because the references were measured with its scikit-learn:

```sh
PYTHONPATH=. ~/coding/OscillateLabs/intent-eval/.venv-bench/bin/python scripts/check_harness.py ~/coding/OscillateLabs/intent-eval
```

Python 3.11–3.14. Targets OVOS with PIPELINE-1 / INTENT-4 (ovos-core 3.7+).

## License

Apache-2.0
