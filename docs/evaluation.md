# Evaluation

Every number here was measured on Switchboard's evaluation harness, which keeps every experiment, including the
failures. Choices were made on a dev split and reported once on a held-out test split. The data, code and results
are public: [switchboard-eval](https://github.com/OscillateLabsLLC/switchboard-eval).

## The data

- **Real traffic** from one household's two devices: a Neon smart speaker and an OVOS desktop. About 65 real
  requests per device per split, so differences under about 6 points are noise. Comparisons use a paired test
  (McNemar).
- **Declared phrases:** requests that match a phrase an installed skill declares word for word.
- **Non-commands:** near misses, off-topic speech, the assistant's own voice heard back, and background talk. Plus
  600 public sentences from meetings, idioms and dialogue, as a stress test.
- **The comparison** is stock OVOS as shipped (Padatious and Model2Vec), not a tuned setup.
- **Hardware:** an Apple M3 Max laptop, a 24 GB GPU (RTX 3090) for the 14B LLM, and a 4-core mini PC for latency.

## Headline

Test split, default settings, with a 14B LLM on a GPU:

| | stock OVOS | Switchboard |
|---|---|---|
| Declared phrases handled correctly | 96.1% | 98.1% (speaker), 100% (desktop) |
| Real speaker requests handled correctly | 42.2% | **84.4%** (p = 5e-7) |
| Real desktop requests handled correctly | 71.4% | **88.9%** (p = 0.007) |
| Acts on non-commands | 100% | **12.5%** |
| High-risk actions run by mistake | 0 | 0 (always held) |

- **Same answer every time:** three fresh builds made identical decisions, and so did the LLM at temperature 0.
- **Fast without an LLM:** 1.9 ms at p95 on the M3 Max, 9.5 ms on the mini PC. Live, end to end, decisions took
  46-108 ms, against 57-114 ms for stock Padatious. These need an ovos-workshop fix
  ([#714](https://github.com/OpenVoiceOS/ovos-workshop/pull/714)); without it every pipeline, stock included, waits
  an extra 0.5 s.
- **In a live trial on the desktop:** 1 wrong route per round of 14 requests; stock had 5.

## How it fails

Real requests on the test split:

| | right | wrong skill | did nothing |
|---|---|---|---|
| stock OVOS, speaker | 42.2% | **56.2%** | 1.6% |
| stock OVOS, desktop | 71.4% | 27.0% | 1.6% |
| Switchboard, no LLM, speaker | 76.6% | **1.6%** | 21.9% |
| Switchboard, no LLM, desktop | 79.4% | 7.9% | 12.7% |
| Switchboard + decider, speaker | 79.7% | 1.6% | 18.8% |
| Switchboard + decider, desktop | 82.5% | 7.9% | 9.5% |
| Switchboard + 14B LLM, speaker | 84.4% | 1.6% | 14.1% |
| Switchboard + 14B LLM, desktop | 88.9% | 9.5% | 1.6% |

Stock OVOS fails by doing the wrong thing. Switchboard fails by doing nothing, which the user can just repeat or
rephrase. About a third of the desktop's "wrong" answers are the weather skill answering "what's the weather
tomorrow" with its current-weather handler: the right skill, which the scoring can't credit.

## What it does well

- **Declared phrases always go to the same place.** Exact matching runs first and never guesses between two skills.
- **It stays quiet when speech isn't for it.** Of 71 real overheard utterances it acted on 7; Neon acted on all 71.
  Of the 600 public sentences, it acted on 2.5% (1.8% without the LLM).
- **Rephrasing works.** Without any LLM it still handles 76.6% of real speaker requests, so the classifier does most
  of the work.
- **No skill changes.** Everything comes from what skills already register.
- **Languages with many phrases do well.** Galician and Catalan skills declare many phrases per intent, and both
  score 93.9% on held-out phrases, against 83.0% for English.

## What it does badly

- **Catch-all phrases** like `play {query}`, `say {x}` or `start {application}` claim almost any speech that begins
  with that word. Switchboard lets one step aside only when the classifier is at least 80% sure of another skill,
  and no single threshold fixes every case. Neon's `open {program}` takes "open the weather". Nothing high-risk
  runs, but the wrong skill can.
- **Questions between people.** "Do you think you studied enough?" could be for the assistant or for someone in the
  room; text alone can't tell. By default such questions reach the LLM only within 2 minutes of the assistant
  speaking, so most overheard ones are ignored, but not inside that window.
- **Languages with few phrases.** French skills declare 14 phrases per intent, against 58 for English, so
  Switchboard passes more often in French. In Galician, "not for me" sometimes outranks real requests; the cause
  isn't known yet.
- **Many skills, little memory.** Training runs once, in a separate process, and peaks at about 0.9 GB with 146
  intents and 3.4 GB with 465. Afterwards the model takes 160-240 MB.
- **Not covered:** typed slots and finding the right device stay in the skills. There's no "did you mean X or Y?".

## More skills

Held-out phrases, no LLM:

| skills installed | intents | right | wrong | passed |
|---|---|---|---|---|
| a typical desktop (31 skills) | 146 | 82.9% | 8.7% | 8.4% |
| + all of OpenVoiceOS's skills | 222 | 81.0% | 8.8% | 10.2% |
| + OVOSHatchery | 459 | 68.8% | 17.5% | 13.7% |

More skills mostly means more caution: it passes more and guesses less. Most of the extra wrong answers come from one
phrase, the launcher's `(launch|start|open|run) {application}`, and the test overstates even those (it removes the
other skill's phrase, which normally would collide and go to the classifier). Latency stays around 1-2 ms.

## Choosing a setup

**Recommended:** the LLM step, with a model on a small GPU or an Apple-silicon Mac.

| situation | setup | result (test) |
|---|---|---|
| **No LLM** (CPU only) | leave out the LLM step; `route_decline_class: never` | speaker 76.6%, desktop 79.4%; the most cautious |
| **No LLM, decider** (experimental) | the decider step on the CPU, about 80 ms per decision | speaker 79.7%, desktop 82.5%; the gain over no LLM isn't significant yet |
| **Small local LLM** | Qwen3-1.7B on LM Studio or similar | speaker 81.2%, desktop 85.7%; about 1 s per LLM decision on a Mac's GPU |
| **GPU LLM server** | Qwen3-14B-AWQ on vLLM (24 GB GPU) | speaker 84.4%, desktop 88.9%; about 0.4 s per LLM decision |

- **Bigger isn't automatically better.** Qwen3-4B acted on more non-commands than the 1.7B, was twice as slow, and
  didn't repeat its own answers. A 35B model matched the 14B but didn't fit beside OVOS on a 36 GB machine.
- **Don't run the LLM on a CPU.** A route prompt is about 1,750 tokens, which takes about 8 s on a fast laptop's
  CPU and longer on a mini PC. Without a GPU, use no LLM or the decider.
- **The `always` policy** (send every "not for me" to the LLM) is more accurate (speaker 90.6%) but acts on more
  overheard speech, so it's an option, not the default.
- **Other languages** need their skills' own phrases; set `lang`. The prompts are English for now.

## Not yet measured

- A Raspberry Pi.
- Real speech in any language but English, and test phrases reviewed by native speakers.
- More than one home.
