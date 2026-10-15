# Configuration

## The steps

| pipeline id | what it decides | outcome |
|---|---|---|
| `ovos-switchboard-echo-pipeline-plugin` | the utterance repeats 4+ words in a row the assistant said in the last 60 s | decline |
| `ovos-switchboard-exact-pipeline-plugin` | the whole utterance matches exactly one skill's phrase (after a leading "Hey Mycroft") | dispatch |
| `ovos-switchboard-classifier-pipeline-plugin` | a character n-gram classifier with a "not for me" class | dispatch at confidence 0.5+; decline or pass for "not for me"; pass otherwise |
| `ovos-switchboard-llm-pipeline-plugin` | an LLM picks one of 12 shortlisted skills, then a gate checks the speech was addressed to the assistant | dispatch; decline when it's not for the assistant or the LLM is unreachable; pass questions on to common_qa |
| `ovos-switchboard-prefix-pipeline-plugin` (optional) | the utterance starts with a skill's phrase and keeps going ("make me laugh hey dad …") | dispatch |
| `ovos-switchboard-decider-pipeline-plugin` (experimental) | a small local model picks a shortlisted skill, for hosts without an LLM | dispatch at probability 0.7+; otherwise decline or pass |

- **Dispatch** sends the request to `skill_id:intent_name` with the data that skill expects.
- **Decline** sends it to Hangup, which records it and says nothing. Hangup is required.
- **Pass** leaves it to the next step.

## Pipeline order

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

Why this order:

- **Echo first**, so the assistant's own voice can't trigger stop or converse.
- **OCP after the classifier.** Switchboard hands OCP's own intents to OCP's step, which adds data its handlers
  need. Placed earlier, OCP grabbed phrases meant for other skills ("play something like glados" played music
  instead of the easter egg).
- **The LLM before common_qa**, so overheard questions are declined instead of answered. Questions the LLM
  recognizes as addressed to the assistant pass through to common_qa.
- **The stock matchers are blacklisted**, so ovos-core doesn't load or train them. ovos-core loads every installed
  pipeline plugin even if it's not in the pipeline; on the test desktop, blacklisting cut a restart from 51 s to
  30 s. A stock matcher left after Switchboard gets everything Switchboard passes on, including overheard speech.
- **Without an LLM**, leave out the LLM step and set the classifier's `route_decline_class` to `never`, so "not for
  me" speech is declined instead of reaching common_qa.
- **The decider** (experimental) goes where the LLM step goes. Install it with
  `pip install "ovos-switchboard-pipeline-plugin[decider]"` (brings PyTorch). On the harness it closed about half
  the gap between no LLM and a small LLM, a gain too small to be sure of yet. An LLM on a small GPU or an
  Apple-silicon Mac is still the recommendation.
- **The prefix step** fixed 2 real rows with no losses, but those rows are why it was written, so it isn't part of
  the measured order yet. If you use it, put it between the classifier and the LLM.
- **Any subset works.** The exact step alone is a deterministic replacement for Padatious-high. Results then differ
  from the measured ones.

After changing `intents.pipeline`, restart the listener as well as ovos-core: OVOS sessions keep their own copy of
the pipeline.

## Shadow and active mode

In shadow mode (the default) every step logs what it would have done and how long it took, then lets the existing
pipeline carry on, so nothing changes for the user. OVOS's actual outcome goes to the same log for comparison.
Shadow mode only sees what earlier steps leave, so list the Switchboard steps ahead of the stock matchers. Set
`"mode": "active"` under `intents.switchboard` to switch over.

## Readiness

Switchboard builds its models after all skills have loaded, so OVOS can announce "I am ready" a few seconds before
Switchboard is. To make the announcement wait, add `"switchboard"` to the boot-finished skill's `ready_settings`
(`~/.config/mycroft/skills/ovos-skill-boot-finished.openvoiceos/settings.json`):

```json
{"ready_settings": ["skills", "switchboard"]}
```

## Settings

Shared settings go under `intents.switchboard`; each step's settings go under its pipeline id, and a step can
override the shared `mode`.

| section | key | default | meaning |
|---|---|---|---|
| shared | `mode` | `shadow` | `shadow` logs and never acts; `active` acts |
| shared | `lang` | `en-us` | the language served; other languages pass |
| shared | `hangup_skill_id` | `ovos-skill-hangup.oscillatelabsllc` | where declines go |
| shared | `high_risk` | `defer` | `defer` (hold silently), `confirm` (ask through Doublecheck) or `allow` (testing only) |
| shared | `confirm_skill_id` | `ovos-skill-doublecheck.oscillatelabsllc` | where held requests go with `confirm` |
| shared | `retrain_debounce_s` | 5 | quiet time after skills change before the models rebuild |
| shared | `delegate_skills` | `["ocp"]` | skills whose own pipeline step must handle them |
| shared | `wake_phrases` | the listener's active hotwords | stripped from the start of an utterance before matching |
| shared | `bus_timeout_s` | 10 | each startup query to ovos-core |
| shared | `echo_window_s` | 60 | how recent the assistant's speech must be to count as an echo |
| shared | `speech_history_s`, `speech_lines_max` | 600, 64 | how much of the assistant's speech is kept per session |
| shared | `record_ttl_s`, `records_max` | 30, 64 | per-utterance bookkeeping; keep `record_ttl_s` above the longest pipeline round |
| shared | `model_cache`, `model_cache_path`, `model_cache_max` | on, `$XDG_CACHE_HOME/switchboard`, 2 | reuse the trained model across restarts; stored as plain arrays, so a tampered file can't run code |
| shared | `decision_log`, `decision_log_path` | on, `$XDG_STATE_HOME/switchboard/decisions.jsonl` | every decision and OVOS's actual outcome. Holds household speech; stays local |
| echo | `echo_run_min` | 4 | words in a row shared with recent assistant speech |
| exact | `catch_all_yield` | 0.8 | how confident the classifier must be of another skill to overrule a one-word catch-all phrase such as `play {query}`; 0 never overrules |
| exact | `collision_priority`, `core_skills` | `none`, `[".openvoiceos", ".neongeckocom", "ocp"]` | when two skills declare the same phrase: `none` lets the classifier decide; `core` or `user` lets the one core (or non-core) skill win |
| exact | `skill_priority` | `[]` | an ordered list of skills that win collisions, checked first. Example: `["skill-easter-eggs.openvoiceos", "skill-homeassistant.oscillatelabsllc"]` sends "turn off the lights" to Home Assistant and keeps "turn on grandma mode" for the easter egg |
| exact | `collision_resolution` | `classifier` | when a collision is still undecided after the two lists above: `classifier` lets the classifier decide; `specific` lets the skill whose phrase matched more of the words literally win ("change my alarm to repeat every monday" goes to the repeat intent, not "change my {kind} to {time}"), and a tie still goes to the classifier |
| prefix | `prefix_min_literal` | 3 | words a phrase needs to match as a prefix |
| classifier | `threshold` | 0.5 | confidence needed to dispatch |
| classifier | `route_decline_class` | `active` | what "not for me" does: `active` passes it to the LLM only if the assistant spoke in the last `active_s`; `always` always passes; `never` declines |
| classifier | `active_s` | 120 | the "assistant just spoke" window, in seconds |
| classifier | `dialog_negatives` | true | train "not for me" on the skills' own spoken lines |
| llm | `url`, `model`, `api_key` | none | an OpenAI-compatible endpoint; without `url` the step never matches |
| llm | `timeout_s` | 4 | per request; there are two per utterance |
| llm | `route_max_tokens`, `cache_size` | 200, 1024 | room for one tool call; cached responses |
| llm | `gate`, `shortlist_k` | 0.9, 12 | how sure the gate must be that speech was addressed to the assistant; skills offered per utterance |
| decider | `model`, `revision`, `device` | `tasksource/tasksource-decider-nano`, `v4`, `cpu` | the decision model, downloaded on first use |
| decider | `threshold`, `shortlist_k`, `on_unsure` | 0.7, 12, `decline` | probability needed to act; skills offered; `decline` or `pass` when unsure |

ovos-core gives up on a step that runs past `intents.<pipeline id>.match_timeout` (10 s by default). Keep twice the
LLM's `timeout_s` under that, or a slow LLM falls through to common_qa instead of declining.
