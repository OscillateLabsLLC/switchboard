# Switchboard

Switchboard decides which OVOS skill should handle what someone says, and stays silent when the speech wasn't
meant for the assistant. It replaces OVOS's intent matchers (Padatious, Adapt, Model2Vec) with a few steps that
each had to prove themselves on real household traffic before they were kept.

## Why

Measured on held-out traffic from a Neon smart speaker and an OVOS desktop ([details](docs/evaluation.md)):

- **Real requests handled correctly:** 84% on the speaker and 89% on the desktop, against 42% and 71% for stock
  OVOS.
- **When it's wrong, it usually does nothing** instead of the wrong thing. Stock OVOS ran the wrong skill for 56%
  of the speaker's requests; Switchboard, for 2%.
- **Overheard speech:** of 71 real utterances not meant for the assistant, it acted on 7. Neon acted on all 71.

All of it can be checked: the data, code and results are in
[switchboard-eval](https://github.com/OscillateLabsLLC/switchboard-eval).

## How it works

Each step is a separate OVOS pipeline plugin, tried in this order:

1. **Echo:** ignore the assistant hearing its own voice.
2. **Exact:** if the words match a phrase a skill declares, send it there. No guessing.
3. **Classifier:** a small, fast model trained on the installed skills' phrases picks the skill, or decides the
   speech is "not for me".
4. **LLM** (optional, needs a GPU host): for what's left, a language model chooses among the most likely skills,
   and a second check asks whether the speech was meant for the assistant at all.

Speech that isn't for the assistant goes to [Hangup](https://github.com/OscillateLabsLLC/hangup), a skill that
quietly records it and says nothing. High-risk requests (unlock, open the garage, shut down, clear data) never run
directly: they're held, or with [Doublecheck](https://github.com/OscillateLabsLLC/doublecheck) installed, the
assistant asks first.

## Install

```sh
pip install ovos-switchboard-pipeline-plugin ovos-skill-hangup
```

Then put the steps in `mycroft.conf`. Switchboard starts in **shadow mode**: it logs what it would do and changes
nothing, so you can compare it with your current setup first. Set `"mode": "active"` when you're ready.

```json
"intents": {
  "switchboard": {"mode": "shadow"},
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
  "ovos-switchboard-llm-pipeline-plugin": {
    "url": "https://your-llm-server/v1/chat/completions", "model": "Qwen/Qwen3-14B-AWQ"
  }
}
```

The LLM step works with any OpenAI-compatible chat completions endpoint (vLLM, LM Studio, Ollama, a hosted
service); add `"api_key"` if yours needs one. Restart ovos-core and the listener afterwards. No LLM? Leave out the LLM step and its settings; Switchboard still
beats stock OVOS without one. [docs/configuration.md](docs/configuration.md) explains the order, every setting,
and which stock matchers to switch off.

**Optional: ask before risky requests.** By default, high-risk requests (unlock, open the garage, shut down, clear
data) are held silently. To have the assistant ask first, install Doublecheck and turn it on:

```sh
pip install ovos-skill-doublecheck
```

```json
"intents": {"switchboard": {"high_risk": "confirm"}}
```

## Limitations

- Phrases like `play {anything}` or `open {anything}` can claim speech meant for another skill.
- When two skills could fit, it picks one; it never asks "did you mean X or Y?" (Doublecheck only confirms risky requests).
- Questions between people ("do you think it'll rain?") can be answered if the assistant spoke in the last two
  minutes.
- Only English has been measured on real speech. The LLM prompts are English.
- Measured in one home, on a laptop and a mini PC. Not yet on a Raspberry Pi.

More, with numbers: [docs/evaluation.md](docs/evaluation.md).

## More

- [docs/evaluation.md](docs/evaluation.md): what it does well and badly, and which setup to choose.
- [docs/configuration.md](docs/configuration.md): pipeline order, every setting, shadow mode.
- [docs/design.md](docs/design.md): where the design came from, and how the code is laid out.
- [docs/confirm-flow.md](docs/confirm-flow.md): how confirming high-risk requests works.
- [docs/dynamic-schema.md](docs/dynamic-schema.md): what's planned for skills the schema doesn't know.

## Development

```sh
uv venv && uv pip install -e ".[test,dev]"
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/pylint ovos_switchboard scripts
```

`scripts/check_harness.py <switchboard-eval checkout>` checks that this package still makes exactly the decisions
that were measured. Python 3.11-3.14, ovos-core 3.7+ (OVOS-PIPELINE-1, INTENT-4).

## License

Apache-2.0
