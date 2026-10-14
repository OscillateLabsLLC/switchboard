# Design: confirming high-risk actions

Status: for review. Nothing here is built. The questions raised in the first draft are answered under Decided.

## Today

High-risk intents (unlock, open covers, shutdown, clear data; `risk: high` in the schema) are never acted on. With `high_risk: "defer"`, a step that would dispatch one declines it through Hangup, with reason `held_high_risk` and the held intent and label recorded. Hangup stays silent, so the user hears nothing and nothing happens. That's safe, but it gives no way to say "yes, I meant it".

## Goal

"Unlock the front door" gets a spoken question. A clear "yes" within a few seconds runs exactly the action that was held. Anything else does nothing:
- "no";
- silence;
- an unclear answer;
- an overheard "yes" later on.

## Proposal

1. **A new, small skill does the asking:** `ovos-skill-switchboard-confirm` (name open). Hangup's contract is silence, so it shouldn't gain a voice. Like Hangup, Switchboard requires this skill when confirmation is on.
2. **Switchboard hands the held action to it** instead of to Hangup. When `high_risk: "confirm"`, the step that would dispatch a high-risk intent returns a match for the confirm skill carrying:
   - the held skill label (`skill_id:intent_name`);
   - the match data the skill would have received (today's hold doesn't pass this; it has to);
   - the utterance and the session.
3. **The confirm skill asks, using the user's own words.** "You said: unlock the front door. Should I go ahead?" The frame is a translatable `.dialog` file, and the quoted utterance needs no translation. This avoids writing a spoken description for every high-risk intent in every language.
4. **It reads the answer with OVOS's own yes/no handling** (`ask_yesno`: localized yes/no vocabularies, one short listening window).
   - **On "yes":** it re-emits the held intent the way ovos-core's intent service does on a match (`ovos_core/intent_services/service.py`):
     - a reply to the original utterance message, which keeps its context and session;
     - named `skill_id:intent_name`;
     - data = the original message data merged with the match data, plus `utterance` and `lang`;
     - `skill_id` in the context.

     The target skill can't tell the difference. The confirm skill therefore has to carry the original message's data and context, not just the match.
   - **On "no", silence, a timeout or anything unclear:** it says a short "OK, I won't" (`.dialog`) and does nothing.
5. **Fail safe:**
   - the held action is kept only in memory, only for that one question, and is dropped on timeout;
   - a "yes" outside the question window is just an ordinary utterance and goes through the normal pipeline, where "yes" alone dispatches nothing;
   - Switchboard's echo guard already keeps the assistant's own question from answering itself.
6. **Config:** `high_risk` gains `"confirm"`. `"defer"` (silent hold) stays available; `"allow"` stays for testing. Default: `"confirm"` once it's been tried live.

## How it would be tested

- **Unit tests:** in the step, a high-risk verdict becomes a confirm match carrying label, match data and session. In the skill, yes re-emits exactly the held message, and no, timeout and unclear emit nothing.
- **Live, without ever saying a real high-risk phrase** (standing rule): a stand-in intent marked `risk: high` in a test schema. For example, a harmless intent such as a timer, temporarily treated as high-risk on the test Mac, then the full spoken round trip.
- **The harness** scores routing, not conversations. Its numbers don't change: a held row is still "not executed".

## Decided

1. **Skill name and repo:** a new OscillateLabsLLC skill in its own repo, like Hangup. The name is still open.
2. **Default:** `"confirm"`, but only after a live trial. Until then, `"defer"` stays the default.
3. **Wording:** quote the user ("You said: … Should I go ahead?"). Quoting needs no per-intent text and survives ASR errors honestly, because the user hears what was actually understood. A per-intent description ("Unlock the front door?") would read more naturally but needs maintenance per intent and language.
4. **Medium-risk intents** (e.g. `alarm.cancel`): unchanged, never asked about.
