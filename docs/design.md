# Design

Switchboard started as a question: how did commercial assistants route requests before LLMs? The answer produced a
design, and testing against real household traffic cut it down to what's here.

## What it borrowed

- **Alexa's two layers.** Alexa compiled each skill's sample phrases into an exact grammar, plus a statistical model
  for everything else. Switchboard keeps that split: the exact step, then the classifier, both trained on what the
  skills already declare.
- **Alexa's shortlist-then-rerank.** Amazon narrowed many skills to a short list, then reranked it using context.
  Switchboard's LLM step does the same: a shortlist of 12 skills, the LLM to choose, and "did the assistant just
  speak?" as context.
- **Google's schema-guided approach.** Describe intents in plain language, so a model can handle skills it was never
  trained on. Switchboard's LLM tools are generated from each skill's registrations, so a new skill needs nothing
  written by hand.
- **Google's search fallback.** Questions the assistant can't route go to OVOS's common_qa instead of a dead end.
- **Separate finding, ranking and deciding.** Padatious folds all three into one confidence score that isn't
  comparable across intents, which is why it confidently picks the wrong one. Keeping them apart is also why
  high-risk requests are held by rule rather than trusted to a threshold: wrong high-risk picks reached 0.994
  confidence.
- **No skill changes.** Tools are named `skill_id:intent_name` and use the skill's own slot names, so a skill can't
  tell Switchboard's dispatch from a normal match.

## What testing changed

The first plan had seven stages, including a sentence encoder, a slot-filling model and a trained reranker. Each
stage had to outperform Adapt, Padatious and Model2Vec before the next one was built.

- **No sentence encoder.** A character n-gram classifier tied a small sentence encoder and outperformed Model2Vec
  model. It needs no GPU and no download, and decides in about 10 ms at p95 on a 4-core mini PC.
- **Exact matching first.** A classifier alone got 76% of skills' own declared phrases right; exact matching first
  got 98.9%. When two skills declare the same phrase, the classifier decides instead of guessing.
- **Knowing when not to act became its own stage.** After the first two, acting on speech that wasn't for the
  assistant was the biggest problem. No open dataset of that exists, so the "not for me" class trains on the
  assistant's own spoken lines, which its microphone is most likely to hear back.
- **The shortlist is required.** All 149 skills on the test desktop came to about 12,000 prompt tokens, more than the
  local model's context.
- **Context from real logs.** Speech was meant for the assistant 87% of the time when it had spoken in the last 30
  s, and 49% otherwise. That became the "assistant just spoke" rule. Requests almost never repeat four words in a
  row of the assistant's speech; its own echo does. That became the echo step.
- **Tried and dropped:** a "clarify" question (on overheard speech it interrupted 86 times and helped 0), and giving
  the LLM the conversation so far (it doubled actions on overheard speech).

## Choices made for now

- **No trained reranker** until logs show misroutes it would fix.
- **Slots come from the matched phrase**; typed parsing and finding the right device stay in the skills.
- **Within OVOS:** declines go to a silent skill (Hangup) because OVOS's fallback skills speak; each stage is its own
  pipeline step so people can adopt only some; shadow mode is the default so a device can be compared with itself
  before anything changes.

## Code layout

```
ovos_switchboard/
  core/       no OVOS imports: templates, schema, device data, the stages (each returns dispatch, decline or
              pass), the LLM port, and router.py, which runs the stages in the measured order
  adapters/   the LLM client, the decision log, the bus registry, dialog files
  ovos/       the pipeline steps and the runtime they share
  data/       the intent schema, reviewed synthetic examples, the prompts
```

Each step builds only its own stage, so installing the exact step alone trains no classifier. Stages build once
ovos-core reports its skills loaded, cheapest first: on a desktop with 147 intents the exact and LLM stages are ready
in about 2 s, and the classifier loads from its cache in under 2 s (about 20 s if it must retrain). When skills
change, everything rebuilds in the background.
