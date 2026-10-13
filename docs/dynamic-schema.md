# Dynamic schema: routing arbitrary skill intents

Status: proposal, 2026-10-12. Nothing here is built yet, except the opt-in flag measured below.

## The problem

Switchboard's canonical schema (`data/intents.yaml`) was written by hand for the skills on two devices. On the Mac,
**62 of 146 registered intents (42%) aren't in it.** An unmapped intent still trains the classifier as its own
class and still gets an LLM tool, but the classifier never acts on it. Every request for it pays for an LLM call
(about 1 s) or gets no decision at all without an LLM. Hand-mapping fixes one device at a time
(`intent-eval/mac_mapping_proposals.yaml` drafts the Mac's 62). That doesn't scale to skills nobody here has seen,
which is the normal case for an upstream default.

## What intents.yaml does today

| Job | Used by | For an unmapped intent today |
|---|---|---|
| Groups equivalent intents across skills and devices (Neon's `WhereWereYouBorn` = OVOS's `where_were_you_born`) | classifier classes, exact-match collisions, cross-device phrase sharing | its own class, alone |
| Lets the classifier act | classifier | **never dispatched by the classifier**; the LLM decides |
| Keys the reviewed synthetic examples | classifier training | gets none: synthetic rows carry canonical intents |
| Adds a description per intent | classifier training (one pseudo-example each) | none; the LLM tool's description is derived from the skill's own phrases |
| Renames skill slots to canonical slots | the decision log, the harness | slots stay under the skill's own names (the dispatch is unaffected) |
| Sets risk (high-risk intents are held) | all dispatching steps | treated as **low risk**, whatever it does |

## Evidence so far

- **Live (Mac shadow run):** "where were you born" topped the classifier at 0.959, but the intent is unmapped, so
  it went to the LLM: right answer, plus 1 s.
- **Harness experiment (`intent-eval/exp_unmapped.py`, BENCH.md):** letting the classifier dispatch unmapped intents
  changed 2 utterances across both devices and splits. Both were wrong: "how far is the moon from earth" went to
  date-time `time_until` (0.578), and "how do i unlock my phone" went to Neon's communication `call` (0.676), which
  would place a call. Nothing else moved.
- **The harness can't see the upside.** Gold labels are canonical intents, and an unmapped intent has none, so a
  correct unmapped dispatch can never score as right. Hub replay labels are canonical too. No label-level gold
  exists yet.
- **A lead, not a rule:** both losses scored 0.58 to 0.68, and the live win scored 0.96. A stricter bar for unmapped
  picks might separate them. That's three points, two of them from the test split.

## Design

### 1. Skill intents are the classes; canonical intents are an optional grouping

Every registered `skill_id:intent_name` is a first-class class, which is already true for training. The schema
stops being a gate and becomes an overlay that adds value where it exists: merged classes, curated examples, slot
renames, descriptions, risk. An intent missing from the schema loses those extras, not the ability to be routed.

What changes is the classifier's dispatch rule for unmapped classes. The rule should be **chosen on label-level
data (section 6), not here.** Candidates, in order of preference:
1. Dispatch an unmapped pick only above a stricter threshold than mapped picks (the lead above).
2. Dispatch it only when the classifier's pick and the LLM's tool choice agree. That's safer, but it costs the
   LLM's latency, so it only saves the second LLM call.
3. Keep passing to the LLM (today's behavior), and lean on the overlay and proposals (sections 2 and 3) to shrink
   the unmapped set.

`ClassifierOptions.dispatch_unmapped` (branch `exp/dispatch-unmapped`, off by default) is the switch for candidate 1.

### 2. Schema overlays per instance

Today one file is loaded (the bundled schema, or `schema_path`). Proposed: the bundled schema, plus every
`*.yaml` in an overlay directory (`$XDG_CONFIG_HOME/switchboard/intents.d/` by default), merged in name order. An
overlay can add canonical intents, map labels into existing ones, set risk, add descriptions and decline labels.
That gives each household or distribution a place for mappings without forking the package, and lets proposals
(section 3) land as one reviewable file. Merge conflicts, such as one label mapped to two canonicals, fail loudly
at load.

### 3. Mapping proposals instead of hand-editing

A command (`switchboard propose-mappings`) reads the live registrations, finds unmapped intents and writes a
proposal file in the overlay format, for a human to review. Proposals come from cheap signals first:
- the same intent name in another skill already mapped (Neon's and OVOS's personal skills)
- template overlap with mapped intents
- an optional LLM pass over the skill's phrases and the canonical descriptions

Nothing is applied without review. That's the existing guardrail for label mappings, and tonight's Mac draft is
this process done by hand.

### 4. Training data for intents nobody curated

Synthetic rows gain an optional `label` key alongside `intent`, so reviewed examples can target one skill intent
without inventing a canonical. The same proposal command can draft paraphrases for unmapped intents, for review
like the existing synthetic set. Dialog-derived decline negatives already work for any installed skill.

### 5. Unknown risk

Unmapped intents are treated as low risk today; the experiment's `call` dispatch shows what that costs. Per the
decision to handle risk with the confirm flow, the proposal is only to *classify* risk for unmapped intents, with
keyword heuristics over the intent name and phrases (unlock, open + door/garage, delete/clear/erase, shutdown,
call/text/email/send, buy/order). The confirm flow then acts on that, like any high-risk intent. Until it exists,
heuristically risky unmapped intents should never dispatch from the classifier.

### 6. Evaluation at the skill-intent level

The harness needs gold that can credit an unmapped dispatch:
- Dataset and replay rows gain an optional `labels` field: the set of acceptable `skill_id:intent_name` dispatches
  on each device.
- Scoring accepts a match on either the canonical intent or one of the labels.
- New rows come from the shadow decision log: Switchboard's verdict, OVOS's outcome, and a human label, starting
  with the disagreements.

This blocks section 1's rule, so it comes first.

### 7. Wake phrases in transcripts

A leading wake phrase ("Hey Mycroft, where were you born") cost one exact match live. Measured: Hub 0.3% of
transcripts, Mac 1.5%, and two of the Mac's nine are the assistant's own speech read back. Stripping is built
(PR #5: the exact and prefix steps strip a leading wake phrase). Still proposed: record it as an addressee feature,
since a spoken wake word is evidence the speech was meant for the assistant. Low priority.

## Decided (2026-10-13)

- **Overlays live in an XDG directory** (`$XDG_CONFIG_HOME/switchboard/intents.d/`), not in `mycroft.conf`.
- **Approved mappings for stock OVOS and Neon skills go upstream** into the bundled schema, so default
  installations get them without an overlay. Overlays are for what's specific to one household or distribution.
  Neon's mappings aren't final: they will grow as skills currently blacklisted on the Hub are enabled, so they are
  revisited each time Neon's skill set changes.
- **A canonical intent groups labels that do the same thing** (Neon's `WhereWereYouBorn` and OVOS's
  `where_were_you_born`), never sibling actions of one skill. When siblings share a class, the classifier learns
  one class for all of them and a weaker second step (`LabelIndex.select`: template match, else nearest examples)
  picks the handler, so "how's my CPU doing" can reach the right class and still get the memory handler. A skill
  intent with no cross-skill equivalent gets its own canonical, or stays unmapped with an overlay entry that only
  sets risk and a description.

  Measured (`intent-eval/exp_handler_choice.py`, BENCH.md): 29 of the 70 canonicals already group siblings. On the
  rows that can be judged, the wrong handler was picked once ("enable spicy mode" -> `malibu_stacey_intent`, via
  nearest examples). Exact-stage dispatches pick the handler by template, so the risk is in classifier dispatches,
  where label gold is thin. The Mac mappings follow the rule now; retrofitting the existing 29 waits for phase 1.

## Phases

1. **Label-level evaluation (section 6):** the `labels` field, label-aware scoring, a labeling queue fed by the
   shadow log. No behavior change.
2. **Overlays (section 2):** no behavior change for anyone without an overlay. The reviewed Mac mappings go into the
   bundled schema, not an overlay (see Decided).
3. **Choose the unmapped dispatch rule (section 1)** on phase 1 data. Ship it only if it's never worse on any slice.
4. **Mapping proposals and label-keyed synthetic examples (sections 3 and 4).**
5. **Risk heuristics (section 5),** together with the confirm flow.
6. **Retrofit the grouping rule** onto the existing schema's 29 sibling-grouping canonicals, judged on phase 1's
   label-level gold.
