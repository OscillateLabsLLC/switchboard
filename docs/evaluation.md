# Evaluation: what Switchboard does well, what it doesn't, and how to configure it

Every number here comes from the [intent-eval](https://forgejo.graywind.org/graywind/intent-eval) harness. `BENCH.md` there records each experiment, including the ones that failed. Choices were made on the dev split and reported on the test split once.

## The data, and how far to trust it

- **Real household traffic** from one household: a Neon Hub (the main device) and an OVOS Mac. About 65 real requests per device per split, so differences under about 6 points are noise. Paired comparisons use McNemar's test.
- **Magic phrases:** rows that match a phrase some installed skill declares.
- **Non-commands:** near misses, out-of-domain speech, the assistant's own TTS heard back, and background speech. Plus 600 external sentences from meetings, idioms and dialogue, as a stress test.
- **Comparator:** stock OVOS as shipped (Padatious + Model2Vec), not a hand-tuned setup.
- **Hardware:** an Apple M3 Max, plus an RTX 3090 on the LAN for the 14B LLM. **Nothing has been measured on a mini PC or a Raspberry Pi yet.**
- **Other languages:** English is the only language with real traffic. The others are measured on template hold-outs, plus 62 Galician sentences drafted by Claude and **not reviewed by a native speaker**.

## Headline (test split, default configuration, 14B LLM on the LAN)

| | stock OVOS | Switchboard |
|---|---|---|
| Declared phrases ("magic") right | 96.1% | 98.1% (Hub) / 100% (Mac) |
| Real Hub requests right | 42.2% | **84.4%** (p = 4.6e-7) |
| Real Mac requests right | 69.8% | **87.3%** (p = 0.007) |
| Acts on non-commands | 100% | **12.5%** |
| High-risk actions executed by mistake | 0 | 0 (always held) |

- **Same answer every time.** Three fresh builds gave zero routing differences. A local LLM at temperature 0, rerun without its cache, reproduced every decision.
- **Speed (M3 Max).**
  - The decision without the LLM takes 1.9 ms at p95.
  - Live end-to-end decisions take 46–108 ms. Shipped Padatious takes 57–114 ms; Padacioso, 78–137 ms.
  - These numbers need the ovos-workshop converse fix ([OpenVoiceOS/ovos-workshop#714](https://github.com/OpenVoiceOS/ovos-workshop/pull/714)). Without it, every pipeline, stock included, waits an extra 0.5 s per utterance.
- **Live, on the Mac:** 1 wrong route per 14-utterance round. Both stock pipelines had 5.

The README's earlier 90.6% Hub figure used the opt-in `always` decline policy. That policy fails the false-accept limit on dev (27.3% against 25%), so the default numbers above are the fair ones.

## What it does well

- **Declared phrases route deterministically.** Exact matching runs first and never guesses between colliding templates.
- **It stays quiet when speech isn't for it.**
  - Replaying the Hub's history, it acted on 7 of 71 real overheard utterances; Neon had acted on all 71. That
    replay used `always` routing, the less cautious policy.
  - On 600 external overheard sentences (meetings, idioms, dialogue), treated as false wakes, the default configuration acted on 2.7%. Without the LLM it was 1.8%; with the opt-in `always` policy, 13.8%.
- **Paraphrases are handled.** On real Hub traffic, Switchboard doubles stock's accuracy. Without any LLM it still reaches 76.6%, so the classifier carries most of that gain; the LLM adds the rest.
- **No skill changes are needed.** LLM tools come from what skills already register. Dispatches carry the skill's own slot names.
- **Rich-template languages work well.** On template hold-outs, Galician and Catalan both score 93.9% right, compared with 83.0% for English.

## What it does badly, or not at all

- **Verb + free-slot templates** such as `play {query}`, `say {x}` and `start {application}`.
  - The exact layer treats these as declared phrases. A catch-all steps aside only when the classifier is at least 0.8 confident of something else.
  - Lowering that threshold breaks real "say …" and "play …" commands, so a global threshold can't fix it (`exp_yield_sweep.py`).
  - On the Hub, Neon's launcher template `open {program}` takes "open the garage", "open the blinds" and "open the weather" through the exact layer. Nothing high-risk runs, but the user gets the wrong skill.
  - OVOS's launcher keeps such phrases out with a runtime blacklist. Switchboard can't see that blacklist.
- **Unlisted paraphrases near a catch-all.** "Start counting to ten" goes to the launcher if the count skill doesn't list that phrasing.
- **Questions between people.** "Do you think you studied enough?" is phrased to "you", and text alone can't tell who "you" is. Under `always` routing, 59 of the 83 external sentences acted on were questions like this sent to `ask_knowledge`.
  - The default `active` policy sends that kind of row to the LLM only within 120 s of the assistant speaking, so cold overheard questions mostly stay out (5 LLM accepts of 600). Within that window, the problem remains.
- **Thin-template languages.** French skills declare 14 template lines per intent, against 58 for English and 125 for Galician. French passes 23% of hold-out phrases. It isn't wrong more often; it just covers less.
- **In Galician, the decline class can outrank real requests.** Five of the 7 missed Galician requests had "not for me" on top, three with the right intent second. Why is not yet measured.
- **Memory at scale.**
  - At 465 intents, training holds 3.5 GB in process. The trained model itself is 207 MB once loaded from cache.
  - Training in a child process is planned before Switchboard is recommended to OVOS users.
- **Out of scope.** Slot typing and entity resolution stay in the skills. There is no confirm flow yet: high-risk intents are held silently. There is no disambiguation ("did you mean X or Y?") yet.

## Scale: more skills mostly means more caution

These numbers are from template hold-outs with no LLM: every fifth template line is held out and routed.

| skills installed | intents | right | wrong | passed | train time | peak memory |
|---|---|---|---|---|---|---|
| the Mac's 31 skills | 146 | 83.0% | 8.6% | 8.4% | 10 s | 0.9 GB |
| + OpenVoiceOS org | 222 | 80.7% | 8.8% | 10.5% | 21 s | 1.6 GB |
| + OVOSHatchery | 459 | 72.9% | 13.4% | 13.7% | 59 s | 3.9 GB |
| + OscillateLabsLLC | 465 | 69.0% | 17.5% | 13.5% | 57 s | 4.2 GB |

Peak memory is the experiment process, which builds two stacks: an upper bound, not a device's steady state.

- **Most wrong answers come from one catch-all.** At every size, the largest single source of wrong answers is the OVOS launcher's `(launch|start|open|run) {application}` catching the count skill's lines. Leave those out and wrong answers stay at 2.9–4.0% at every size.
- **This test overstates catch-all harm.** It removes the victim's own template. With every template installed, the two would collide and the classifier would decide. No real household row is a launcher misroute.
- **Real scale effects:**
  - Passes grow (8.4% → 13.7%): more classes dilute confidence, so the assistant declines rather than guesses.
  - Training time and memory grow too.
  - Latency stays around 1–2 ms.

## Choosing a configuration

| Situation | Configuration | What you get (test) |
|---|---|---|
| **No LLM** (CPU only, offline) | Leave out the LLM step; set `route_decline_class` to `never` | Real Hub 76.6% (beats stock, p = 2.7e-5), Mac 77.8% (not significant). Magic 95–97%, below the 98% bar. Most cautious: acts on 7.5–10% of non-commands |
| **Small local LLM** | Qwen3-1.7B (1.84 GB file) on LM Studio or similar, `shortlist_k` 12 | Passes every criterion. Real Hub 81.2%, Mac 84.1%, non-commands 10–12.5%. LLM route p50 965 ms, gate 114 ms (M3 Max Metal) |
| **Small local LLM, latency first** | Same, `shortlist_k` 6 | Route p50 623 ms (-35%). About 1.5 points lower on real traffic, and magic 97.1% on the Hub, below the 98% bar |
| **GPU on the LAN** | Qwen3-14B-AWQ on vLLM (RTX 3090) | Best measured: real Hub 84.4%, Mac 87.3%, non-commands 12.5%. About 400–470 ms per fresh LLM decision |

- **Bigger models aren't automatically better.**
  - Qwen3-4B matched the 1.7B's accuracy but acted on more non-commands. On dev it acted on 33.3%, failing the 25% limit.
    It's also twice as slow (route p50 1,978 ms), and on LM Studio it wasn't repeatable: about 5% of its LLM decisions
    changed between two runs at temperature 0. The 1.7B reproduced every decision.
  - Qwen3.6-35B-A3B scored level with the 14B within noise. At 20.5 GB it doesn't fit beside the OVOS stack on a 36 GB Mac: the run swapped.
- **No GPU: don't run the LLM on CPU.** The route prompt is about 1,750 tokens (each skill tool about 100). On the M3 Max's
  plain CPU cores, Qwen3-1.7B (Q8_0 GGUF) reads prompts at about 224 tokens/s, so about 8 s per route call (5 s with 6 tools).
  A typical mini PC would be several times slower (estimate). Without a GPU, use the no-LLM configuration.
- **The small-model numbers are MLX builds** (8-bit 1.7B, 4-bit 4B) on Apple's GPU. GGUF builds of the same models on other
  hardware weren't checked for identical answers.
- **Servers without logprobs (LM Studio).** The addressee gate reads a plain yes/no, so it's binary rather than calibrated. Servers that don't enforce tool choice let small models invent tool names; Switchboard declines those (`unknown_tool`).
- **Languages.**
  - Set `lang`. Every language needs its skills' own templates, and the decline class trains on that language's `.dialog` lines.
  - Prompts are English for now. With the 1.7B and English prompts, Galician paraphrases scored 75.0% right, 8.3% wrong and 16.7% passed.
- **Decline policy.**
  - `active` (default): send "not for me" verdicts to the LLM only if the assistant spoke in the last 120 s.
  - `always`: higher accuracy (Hub 90.6%) but acts on more overheard speech. Fine for a household that tolerates that, not as a default.
- **Many skills (400+).**
  - Expect more passes, not many more wrong answers, except around catch-all templates.
  - Budget about 4 GB for the first training. Later starts load the cached model.

## Not yet measured

- Mini PC and Raspberry Pi latency and memory.
- Native-speaker evaluation data for any language except English.
- A second household's traffic.
- Training memory with float32 or a different solver.
