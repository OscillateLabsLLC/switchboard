# Roadmap: skills the schema doesn't know

Status: planned. Nothing here is built yet.

## The problem

Switchboard's schema (`ovos_switchboard/data/intents.yaml`) groups equivalent intents across skills ("where were you
born" in Neon's skill and OVOS's), sets each intent's risk, and adds curated examples. It was written by hand for
the skills on two devices, so it doesn't know most skills: on the test desktop, 62 of 146 installed intents (42%)
aren't in it.

An intent missing from the schema still works, but less well:

- the classifier never sends requests to it, so each one waits for the LLM (about 1 s), or gets no answer without
  an LLM;
- it gets no curated training examples;
- it's treated as low risk, whatever it does.

Mapping intents by hand fixes one device at a time. An upstream default has to handle skills nobody here has seen.

## The plan

1. **Score at the skill-intent level first.** Today's test data is labeled with schema intents, so a correct
   decision for an unknown intent can't be counted as right. Test rows gain the acceptable `skill_id:intent_name`
   answers; new rows come from the decision log, starting with disagreements.
2. **Local overlays.** Besides the bundled schema, load every `*.yaml` in `$XDG_CONFIG_HOME/switchboard/intents.d/`,
   so a household or distribution can add mappings without forking. Mappings for stock OVOS and Neon skills go into
   the bundled schema instead, so default installs get them.
3. **Let the classifier send requests to unknown intents**, but only under a rule chosen on step 1's data: a
   stricter confidence bar, or agreement with the LLM. An early experiment found two such dispatches, both wrong
   (at 0.58 and 0.68 confidence), and one right live (at 0.96), which suggests a stricter bar, but that's three
   data points.
4. **Proposed mappings, reviewed by a person.** A command reads the installed skills and drafts schema entries for
   unknown intents, from cheap signals (the same intent name in another skill, overlapping phrases), optionally an
   LLM. Nothing is applied without review.
5. **Risk for unknown intents** from keywords in the intent's name and phrases (unlock, open the door, delete, shut
   down, call, buy), so a risky intent is held or confirmed even without a schema entry.

## Rules already decided

- A schema intent groups intents that do the same thing in different skills, never different actions of one skill.
- Overlays live in an XDG directory, not in `mycroft.conf`.
