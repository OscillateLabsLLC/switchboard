# Confirming high-risk requests

Some requests shouldn't run on a misheard word: unlocking a door, opening the garage, shutting down, clearing data.
These are marked `risk: high` in the schema, and Switchboard never runs them directly. Confidence can't make them
safe: wrong high-risk picks have reached 0.994.

## Two options

- **`high_risk: "defer"` (default):** the request is held silently. It goes to Hangup, which logs it, and nothing
  else happens.
- **`high_risk: "confirm"`:** the request goes to [Doublecheck](https://github.com/OscillateLabsLLC/doublecheck),
  which asks "I think you said: unlock the front door. Should I do that?" A clear yes runs exactly the held request;
  anything else (no, silence, an unclear answer) runs nothing.

Either way, the request is consumed: no later pipeline step can run it.

`defer` stays the default because Switchboard can't guarantee Doublecheck is installed. If confirm is on but
Doublecheck isn't loaded at startup, Switchboard logs an error and holds requests silently instead.

## Why quote the user

Doublecheck repeats what it heard instead of describing the action ("Unlock the front door?"). That needs no
per-intent wording in every language, and it's honest about speech recognition: if the assistant misheard, the
user hears the mistake before anything runs.

## How the yes runs

On a yes, Doublecheck sends the skill exactly the message ovos-core would have sent: the original utterance's data
and context, the matched slots, and the skill's id. The skill can't tell a confirmed request from a normal one.

## Safety

- The held request lives only in memory, for one question.
- A "yes" said later is just an ordinary utterance and runs nothing.
- Doublecheck only asks in languages it can understand the answer in (see its README). Otherwise it holds silently.
- Medium-risk intents, such as cancelling an alarm, aren't asked about.
