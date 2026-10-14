# Switchboard

Intent routing for OVOS as a set of pipeline steps. It tries deterministic exact matches first, then a small classifier, then an LLM with tools derived from the skills' own registrations. When speech wasn't meant for the assistant, it stays silent.

Each stage was measured before it was adopted, on a separate evaluation harness, using real traffic from a Neon smart speaker and an OVOS desktop. On held-out smart-speaker traffic, Switchboard answered 84.4% of requests correctly with its default settings, versus 42.2% for stock OVOS. On 71 real overheard utterances, it acted on 7 where Neon acted on all 71. [docs/evaluation.md](docs/evaluation.md) covers what it does well and badly, other languages, scale, and which configuration to use with or without an LLM. `scripts/check_harness.py` shows that this package's stages reproduce the harness's measured decisions exactly.

## The steps

Each step is its own OVOS pipeline plugin. You can place, drop or adopt each one on its own.

| pipeline id | decides | outcomes |
|---|---|---|
| `ovos-switchboard-echo-pipeline-plugin` | the utterance repeats 4+ contiguous words the assistant said in the last 60 s | decline |
| `ovos-switchboard-exact-pipeline-plugin` | the whole utterance matches exactly one skill's template, after stripping a leading wake phrase ("Hey Mycroft, …") | dispatch |
| `ovos-switchboard-classifier-pipeline-plugin` | char n-gram TF-IDF + logistic regression with a `__decline__` class | dispatch if confidence ≥ 0.5; decline or pass for the decline class (policy); pass otherwise |
| `ovos-switchboard-prefix-pipeline-plugin` (optional) | the utterance starts with a slot-free template of 3+ literal words and keeps going ("make me laugh hey dad …") | dispatch |
| `ovos-switchboard-llm-pipeline-plugin` | an LLM picks one of the shortlisted skill tools; an addressee gate checks P(addressed) ≥ 0.9 | dispatch; decline on `ignore`, a failed gate or an unreachable LLM; pass on `ask_knowledge` / `stop` / `unsupported` |
| `ovos-switchboard-decider-pipeline-plugin` (experimental) | a small local decision model picks one of the shortlisted skill intents, described in plain words, with a probability; no tool arguments | dispatch if the top pick's probability ≥ 0.7; decline (or pass, per `on_unsure`) on "none of these" or below threshold; pass on `ask_knowledge` / `stop` |

- **Dispatch** goes to `skill_id:intent_name` with the data that skill reads: template slot values under the template's own names, Adapt vocab matches (`Location`, ...), or LLM tool arguments. Entity resolution and typed parsing stay in the skills.
- **Decline** dispatches to the [Hangup](https://github.com/OscillateLabsLLC/hangup) skill, which speaks nothing and records the utterance and the reason. Hangup is required.
- **Pass** returns no match, so later pipeline steps decide.

Steps never act on each other's verdicts. Pipeline order already means a step only sees what earlier steps passed on. One measured exception: a **catch-all** template, a slot behind one literal word (`play {query}`, `call {contact}`), claims nearly any speech that starts with its word, so the exact step checks it against the classifier model's ranking. When the classifier ranks a different intent (or "not for me") first with `catch_all_yield` confidence, the exact step passes and logs which catch-all was overruled. That keeps "play something like glados" out of media playback, and Neon's own "call me Neon, I am ..." from placing a call. The classifier owns the decline-class policy (`route_decline_class`), which decides whether a "not for me" verdict declines on the spot or passes on to the LLM:
- `active` (default): passes on only if the assistant spoke within the last 120 s.
- `always`: always passes on.
- `never`: always declines on the spot.

A "not for me" utterance that was passed on and then finds no LLM tool (`unsupported`) is declined, not handed to the stock steps after Switchboard.

**High-risk intents** (unlock, open covers, shutdown, clear data) are never acted on directly. A held utterance is consumed, so neither later Switchboard steps nor the stock steps after them (Adapt, Padatious) can run it:
- `high_risk: "defer"` (default): declined silently through Hangup, with reason `held_high_risk` and the held intent recorded.
- `high_risk: "confirm"`: handed to the [Doublecheck](https://github.com/OscillateLabsLLC/doublecheck) skill, which repeats the request back ("You said: unlock the front door. Should I go ahead?") and runs exactly the held intent only on a clear yes. Requires Doublecheck. Planned as the default once it has been tried live ([design](docs/confirm-flow.md)).

## Pipeline order (recommended)

The Switchboard steps are in the order the harness measured. The harness does not model OCP; its placement after the classifier, and the blacklist, come from a live trial on the desktop.

```json
"intents": {
  "pipeline": [
    "ovos-switchboard-echo-pipeline-plugin",
    "ovos-stop-pipeline-plugin-high",
    "ovos-converse-pipeline-plugin",
    "ovos-switchboard-exact-pipeline-plugin",
    "ovos-switchboard-classifier-pipeline-plugin",
    "ovos-ocp-pipeline-plugin-high",
    "ovos-switchboard-llm-pipeline-plugin",
    "ovos-common-query-pipeline-plugin",
    "ovos-fallback-pipeline-plugin-low"
  ],
  "blacklisted_pipelines": [
    "ovos-padacioso-pipeline-plugin", "ovos-padatious-pipeline-plugin", "ovos-m2v-pipeline",
    "ovos-m2v-prototype-pipeline", "ovos-persona-pipeline-plugin", "ovos-adapt-pipeline-plugin",
    "ovos-adapt-domain-pipeline-plugin", "ovos-adapt-hierarchical-pipeline-plugin"
  ]
}
```

- The **echo step comes first** so the assistant's own voice can't trigger stop or converse.
- **OCP comes after the classifier.** Switchboard hands OCP's own intents (`ocp:play`, ...) to OCP's step, which adds the match data its handlers need (`delegate_skills`). Placed before the classifier, OCP would also grab catch-all phrases the exact step yields: live, "play something like glados" played music instead of the easter egg.
- **Stock matchers Switchboard replaces are blacklisted**, so ovos-core never loads or trains them. Listed or not, ovos-core loads every installed pipeline plugin; on the reference desktop this cut a restart from about 51 s to about 30 s. A stock matcher left in the pipeline after Switchboard gets everything Switchboard passes on, which is how overheard speech leaked to m2v in the first active trial.
- The **LLM step runs before common_qa**. If common_qa ran first, it would answer overheard questions before the addressee gate could decline them. Questions the LLM recognizes as addressed to the assistant (`ask_knowledge`) pass through to common_qa.
- The **decider step is an experimental alternative to the LLM step**, for hosts without one. Install it with the `decider` extra (`pip install "ovos-switchboard-pipeline-plugin[decider]"`, which brings PyTorch); without it the step stays inert. Put it where the LLM step goes. It runs a 150M-parameter model on the CPU (about 80 ms per decision on an M3 Max). On the harness it recovered about half the gap between no LLM and a small local LLM, without acting on more overheard speech, but that gain is a few rows and not yet significant ([docs/evaluation.md](docs/evaluation.md)). Recommended instead: the LLM step with a model on a small GPU or an Apple-silicon Mac. To try both, place the decider before the LLM step with `on_unsure: "pass"`, so it answers confident cases and the LLM gets the rest; that combination hasn't been measured.
- The **prefix step is optional** and not part of the measured order. It fixed 2 real trailing-speech rows on the harness with no losses, but those rows motivated it, so it still needs prospective validation. If you use it, place it between the classifier and LLM steps.
- **Using a subset is supported.** For example, the exact step alone makes a deterministic replacement for Padatious-high. Results then differ from the measured numbers above. Without an LLM step, set the classifier's `route_decline_class` to `never`, so "not for me" rows decline instead of reaching common_qa.

## Readiness

Switchboard builds its stages once all skills have registered, so by default OVOS can announce "I am ready" before Switchboard is (8 s early on the reference desktop). Switchboard answers OVOS's readiness query `mycroft.switchboard.is_ready` once every stage is built. To make the announcement wait for it, add `"switchboard"` to the boot-finished skill's `ready_settings` (`~/.config/mycroft/skills/ovos-skill-boot-finished.openvoiceos/settings.json`):

```json
{"ready_settings": ["skills", "switchboard"]}
```

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
| shared | `high_risk` | `defer` | `defer` (hold silently), `confirm` (ask through Doublecheck) or `allow` (testing only); anything else is a startup error |
| shared | `confirm_skill_id` | `ovos-skill-doublecheck.oscillatelabsllc` | where held high-risk requests go with `high_risk: "confirm"` |
| shared | `retrain_debounce_s` | 5 | quiet time after registration changes before stages rebuild |
| shared | `delegate_skills` | `["ocp"]` | skills whose own pipeline step must dispatch them; a Switchboard verdict for one passes to that step |
| shared | `wake_phrases` | the listener's `hotwords` with `listen: true` | stripped from the start of an utterance before template matching |
| shared | `bus_timeout_s` | 10 | each startup query to ovos-core (intent sync, skills list) |
| shared | `echo_window_s` | 60 | echo compares the utterance against assistant speech this recent |
| shared | `speech_history_s`, `speech_lines_max` | 600, 64 | how much assistant speech is kept per session |
| shared | `record_ttl_s`, `records_max` | 30, 64 | per-utterance bookkeeping (high-risk holds, shadow settlement). Keep `record_ttl_s` above the longest pipeline round |
| shared | `model_cache`, `model_cache_path`, `model_cache_max` | on, `$XDG_CACHE_HOME/switchboard`, 2 | reuse the trained classifier across restarts while its inputs are unchanged. Stored as plain arrays (`.npz`, no pickle), so a tampered file can't run code |
| shared | `decision_log`, `decision_log_path` | on, `$XDG_STATE_HOME/switchboard/decisions.jsonl` | JSONL of every verdict and OVOS's actual outcome. Holds household speech; stays local |
| echo | `echo_run_min` | 4 | contiguous words shared with recent assistant speech |
| exact | `catch_all_yield` | 0.8 | classifier confidence in another intent that overrules a catch-all template; 0 never overrules. Without the classifier step, catch-alls always dispatch |
| exact | `collision_priority`, `core_skills` | `none`, `[".openvoiceos", ".neongeckocom", "ocp"]` | when templates of 2+ intents match: `none` lets the classifier decide; `core` / `user` let the one colliding intent whose skills are all core (or all not core) win. `core_skills` entries starting with `.` are skill-id suffixes, the rest exact ids. Measured: `user` fixes "turn off the lights" (alerts' `turn off {alertkind}` vs Home Assistant) without an LLM but sends "turn on grandma mode" to Home Assistant; `core` only hurt |
| exact | `skill_priority` | `[]` | an ordered list of skills (same entry forms as `core_skills`), checked before `collision_priority`: in a collision, the intent of the earliest listed skill wins. Measured: `["skill-easter-eggs.openvoiceos", "skill-homeassistant.oscillatelabsllc"]` fixes "turn off the lights" and keeps "turn on grandma mode", the only setting that got both |
| prefix | `prefix_min_literal` | 3 | literal words a slot-free template needs to match as a prefix |
| classifier | `threshold` | 0.5 | confidence needed to dispatch |
| classifier | `route_decline_class` | `active` | `never` / `active` / `always` |
| classifier | `active_s` | 120 | "assistant active" window, in seconds |
| classifier | `dialog_negatives` | true | train the decline class on installed skills' `.dialog` lines |
| llm | `url`, `model`, `api_key` | none | OpenAI-compatible endpoint. Without `url` the step never matches |
| llm | `timeout_s` | 4 | per request; the step makes two (route + gate) |
| llm | `route_max_tokens`, `cache_size` | 200, 1024 | room for one tool call; in-memory response cache entries |
| llm | `gate`, `shortlist_k` | 0.9, 12 | addressee threshold; skill tools offered per utterance |
| decider | `model`, `revision`, `device` | `tasksource/tasksource-decider-nano`, `v4`, `cpu` | the decision model (Hugging Face Hub; downloaded on first use: config, tokenizer and weights only) |
| decider | `threshold`, `shortlist_k`, `on_unsure` | 0.7, 12, `decline` | top-pick probability needed to act; skill intents offered; `decline` or `pass` on "none of these" or below threshold |

ovos-core abandons any matcher that runs past `intents.<pipeline id>.match_timeout` (default 10 s) and treats it as no match. Keep `2 × timeout_s` under that value, or a slow LLM falls through to common_qa instead of declining.

OVOS sessions carry their own copy of the pipeline. After changing `intents.pipeline`, restart the listener (and any other client that holds a default session) as well as ovos-core. Otherwise spoken utterances keep using the old pipeline, and the Switchboard steps never run.

Every installed step is instantiated by ovos-core even when it isn't in the pipeline. To switch one off entirely, add its id to `intents.blacklisted_pipelines`.

## Shadow mode

In shadow mode every step logs its verdict and latency and returns no match, so the device behaves exactly as before. A later step skips an utterance an earlier step already settled. The LLM step runs off the pipeline thread. OVOS's actual outcome (`ovos.intent.matched` / `unmatched`) goes to the same log for offline comparison. List the Switchboard steps ahead of the stock matchers, or the stock ones will claim utterances first and Switchboard never sees them.

## Where the design came from

Switchboard started as a question about how commercial assistants did intent routing before LLMs. The design that question produced was then cut down to whatever survived measurement against real household traffic.

### What it borrowed

- **Alexa's two-layer NLU.** Alexa compiled each skill's interaction model into a deterministic grammar for exact and near-exact matches, plus a statistical model that generalized beyond them. Switchboard keeps the same split: the exact step, then the classifier. Here, a skill's own registrations are the training data, just as the interaction model was Alexa's.
- **Alexa's arbitration (Shortlister and HypRank).** Amazon's published approach narrows hundreds of thousands of skills to a top-k shortlist, then reranks the candidates using context. Switchboard's LLM step has the same shape: a similarity shortlist of 12 skill tools, and the LLM as the reranker of last resort over that short list. Addressee context (did the assistant just speak?) plays the role of HypRank's context features.
- **Google's schema-guided approach.** Google's Schema-Guided Dialogue work (2019) described intents in natural language, so models could handle services they were never trained on. In Switchboard, the canonical schema's intent descriptions train the classifier, and the LLM's tool descriptions are generated from registrations, so a new skill needs no hand-written schema.
- **Google's search fallback.** When intent matching failed, Google Assistant answered from the web instead of giving up. Here, the `ask_knowledge` meta tool passes knowledge questions to OVOS's common_qa.
- **A clean-slate principle:** keep recall (find candidates), precision (rank them) and policy (act, confirm, or decline) separate. Padatious fuses all three into one confidence score that isn't calibrated across intents. That's why it confidently grabs the wrong intent. Risk-scaled policy is why high-risk intents are held rather than trusted to a threshold.
- **Tools derived from what skills already declare.** The LLM's tools come from INTENT-4 registrations, so no skill has to change. The tool name is `skill_id:intent_name`, and arguments use the skill's own slot names, so the handler can't tell the dispatch apart from a normal match.

### What testing inside OVOS changed

The original plan was seven stages, including a shared sentence encoder, GLiNER slot filling and a trained reranker. Each stage had to beat stock OVOS on the evaluation harness before the next one started:

- **No sentence encoder.** bge-small and a character n-gram TF-IDF classifier tied (p = 1.0). OVOS's own Model2Vec model lost with the same classifier head (p = 0.008). The char n-gram model won on cost: it's CPU-only and needs no model download. Mini PC latency hasn't been measured yet.
- **Exact matching has to come first.** A classifier alone scored 76% on phrases skills declare verbatim; exact-first scored 98.9%. When templates from two intents collide, the phrase goes to the classifier rather than being guessed.
- **Knowing when not to act became its own stage.** After the first two layers, false accepts were the main gap: 71.4%, against stock's 100%. No open "device-directed speech" dataset exists; Amazon's and Apple's corpora are proprietary. So the decline class trains on the assistant's own `.dialog` lines, the sentences its microphone is most likely to pick back up.
- **The shortlist is mandatory, not an optimization.** All 149 tools on the desktop came to about 12.1k prompt tokens, over the local model's 12,288-token context.
- **The addressee gate** reads P(yes) from first-token logprobs, a local stand-in for calibrated cloud addressee models.
- **Context rules came from real smart-speaker logs.** Speech was addressed to the assistant 87% of the time when it had spoken in the last 30 s, and 49% of the time otherwise. That led to the `active` decline policy (pass to the LLM only within 120 s). Requests almost never repeat four or more contiguous words of the assistant's speech (0 of 100), but self-echo does; that became the echo rule.
- **Rejected after measurement:**
  - A `clarify` tool: on overheard speech it interrupted 86 times and recovered 0, because the fragments that confuse the LLM are the same ones that mark speech as overheard.
  - Giving the LLM conversation context: it doubled actions on overheard speech, from 7 to 14 of 71.

### Pragmatic choices

- **No trained reranker yet.** The shortlist plus the gated LLM covers the cases a reranker would decide, and no logs have shown misroutes that need one. It stays on the list, to be earned with data.
- **Template slot extraction instead of GLiNER.** Most slots come straight out of the matched template; typed parsing and entity resolution stay in the skills.
- **High-risk intents are held or confirmed**, rather than trusted to a confidence threshold. Wrong high-risk picks reached 0.994 confidence, so no threshold could separate them.
- **Within OVOS:**
  - The LLM step runs ahead of common_qa so overheard questions are declined, not answered.
  - The echo step runs ahead of stop and converse.
  - Declines go to a silent skill (Hangup), because OVOS's fallback skills speak.
  - Each stage is its own pipeline step, so adopters can take only the pieces they want.
  - Shadow mode is the default, so a device can be compared against its own behavior before anything changes.

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

Each step asks the runtime to build only its own stage, so installing the exact step alone trains no classifier. Stages build once ovos-core reports its skills loaded, cheapest first: on a desktop with 147 intents the exact and LLM stages are ready within about 2 s, and the classifier loads from the model cache in under 2 s (about 20 s when it has to retrain). All stages rebuild from the live INTENT-4 registrations in the background when skills register or deregister.

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
