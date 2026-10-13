# Switchboard

An OVOS intent pipeline plugin. It tries deterministic exact matches first, then a small classifier, then an LLM with tools derived from the skills' own registrations. When speech wasn't meant for the assistant, it stays silent.

Each stage was measured before it was adopted, on the [intent-eval](https://forgejo.graywind.org/graywind/intent-eval) harness, using real household traffic from a Neon Hub and an OVOS Mac. On held-out Hub traffic, Switchboard answered 90.6% of requests correctly versus 42.2% for stock OVOS. On overheard speech, it acted on 7 of 71 utterances where Neon acted on all 71. `scripts/check_harness.py` shows that this package reproduces the harness's measured decisions exactly.

## How it routes

| tier | stage | outcome |
|---|---|---|
| high | echo guard: the utterance repeats 4+ contiguous words the assistant said in the last 60 s | decline |
| high | exact layer: a phrase that matches exactly one skill's template | dispatch |
| medium | classifier (char n-gram TF-IDF + logistic regression, with a `__decline__` class), confidence ≥ 0.5 | dispatch |
| medium | the decline class wins and the policy keeps the row from the LLM | decline |
| low | LLM picks a skill tool and the addressee gate passes (P(addressed) ≥ 0.9) | dispatch |
| low | LLM picks `ignore`, the gate fails, or the LLM is unreachable | decline |
| low | LLM picks `ask_knowledge`, `stop` or `unsupported` | no match: OVOS's common_qa / fallback answer |

A **decline** dispatches to the [Hangup](https://github.com/OscillateLabsLLC/hangup) skill. Hangup speaks nothing and records the utterance and the reason. Hangup is required.

**Decline-class policy** (`route_decline_class`) sets what happens when the classifier says "not for me":
- `active` (default): such rows go to the LLM only if the assistant spoke within the last 120 s.
- `always`: they always go to the LLM.
- `never`: they never do.

**Dispatch** goes to `skill_id:intent_name` with the data that skill reads:
- template slot values under the template's own names
- Adapt vocab matches (`Location`, ...) for keyword intents
- tool arguments for LLM picks

Entity resolution and typed parsing stay in the skills.

**High-risk intents** (unlock, open covers, shutdown, clear data) are never acted on until a confirm flow exists (`high_risk: "defer"`).

The router trains itself from the live INTENT-4 registrations and retrains in the background when skills register or deregister. Its training data:
- the skills' templates and keyword combinations
- the bundled reviewed synthetic examples (`ovos_switchboard/data/`)
- decline negatives harvested from installed skills' `.dialog` files, since the assistant can hear its own voice

## Pipeline order

```json
"intents": {
  "pipeline": [
    "ovos-stop-pipeline-plugin-high",
    "ovos-converse-pipeline-plugin",
    "ovos-switchboard-pipeline-plugin-high",
    "ovos-ocp-pipeline-plugin-high",
    "ovos-switchboard-pipeline-plugin-medium",
    "ovos-switchboard-pipeline-plugin-low",
    "ovos-common-query-pipeline-plugin",
    "ovos-fallback-pipeline-plugin-low"
  ]
}
```

`switchboard-low` runs **before** common_qa. If common_qa ran first, it would answer overheard questions before the LLM's addressee gate could decline them. Questions the LLM recognizes as addressed to the assistant (`ask_knowledge`) fall through to common_qa.

## Configuration

Under `intents.ovos-switchboard-pipeline-plugin` in `mycroft.conf`:

```json
{
  "mode": "shadow",
  "llm": {"url": "http://llm-host:8080/v1/chat/completions", "model": "Qwen/Qwen3-14B-AWQ", "timeout_s": 10},
  "route_decline_class": "active"
}
```

| key | default | meaning |
|---|---|---|
| `mode` | `shadow` | `shadow` logs decisions and never matches; `active` acts |
| `llm` | `{}` | OpenAI-compatible endpoint (`url`, `model`, `api_key`, `timeout_s`). Without it, low-confidence rows fall through to OVOS's own stages |
| `route_decline_class` | `active` | `never` / `active` / `always` |
| `threshold` | 0.5 | classifier confidence needed to dispatch |
| `gate` | 0.9 | addressee-gate P(addressed) needed to act on an LLM pick |
| `shortlist_k` | 12 | skill tools offered to the LLM per utterance |
| `echo_run_min` / `active_s` | 4 / 120 | echo guard run length; "assistant active" window, in seconds |
| `hangup_skill_id` | `ovos-skill-hangup.oscillatelabsllc` | where declines go |
| `high_risk` | `defer` | `defer` or `allow` |
| `decision_log` / `decision_log_path` | on / `$XDG_STATE_HOME/switchboard/decisions.jsonl` | JSONL of every decision and OVOS's actual outcome. Holds household speech; stays local |

## Shadow mode

In shadow mode, Switchboard decides every utterance at whichever of its tiers runs first, then logs the decision and the fast-path latency. LLM calls run off the pipeline thread. It never returns a match, so the device behaves exactly as before. OVOS's actual outcome (`ovos.intent.matched` / `unmatched`) goes to the same log, so the two can be compared offline.

## Development

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test,dev]"
.venv/bin/pytest && .venv/bin/ruff check .
```

The regression gate needs the intent-eval checkout and its environment, because the references were measured with its scikit-learn:

```sh
PYTHONPATH=. ~/coding/OscillateLabs/intent-eval/.venv-bench/bin/python scripts/check_harness.py ~/coding/OscillateLabs/intent-eval
```

Python 3.11–3.14. Targets OVOS with PIPELINE-1 / INTENT-4 (ovos-core 3.7+).

## License

Apache-2.0
