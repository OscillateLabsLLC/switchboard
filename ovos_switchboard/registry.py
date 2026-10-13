"""The device's live intent manifest, indexed from INTENT-4 registration broadcasts.

Same source as ovos-core's IntentManifest (`ovos.intent.register.template` / `.keyword` and their deregistrations),
kept locally so retraining never waits on a bus round trip. A one-time `ovos.intent.list` / `describe` sync covers
registrations broadcast before this plugin loaded (a pipeline reload). Only the default session's registrations
train the router; per-session intents are out of scope.

OCP's own templates are pipeline-private (never broadcast), so they are read from the installed ocp_pipeline package,
under the labels OCP dispatches.
"""
import os
import threading

from ovos_switchboard.device import adapt_definition

DEFAULT_SESSION = "default"
SYNC_TIMEOUT_S = 10
# ocp_pipeline .intent file -> intent name OCP dispatches; media_stop dispatches as ovos.common_play:stop
OCP_INTENTS = {"play.intent": "play", "pause.intent": "pause", "next.intent": "next",
               "prev.intent": "prev", "resume.intent": "resume", "media_stop.intent": "stop"}
OCP_SKILL_IDS = {"stop": "ovos.common_play"}


def session_of(message):
    sess = message.context.get("session")
    return sess.get("session_id", DEFAULT_SESSION) if isinstance(sess, dict) else DEFAULT_SESSION


class IntentRegistry:
    def __init__(self, bus, on_change=lambda: None):
        self.bus, self.on_change = bus, on_change
        self.lock = threading.Lock()
        self.entries = {}  # (skill_id, intent_name, lang, method) -> {"enabled": bool, "definition": payload}
        self.handlers = {
            "ovos.intent.register.template": self._on_register,
            "ovos.intent.register.keyword": self._on_register,
            "ovos.intent.deregister": self._on_deregister,
            "ovos.skill.deregister": self._on_skill_deregister,
            "ovos.intent.enable": self._on_enable_disable,
            "ovos.intent.disable": self._on_enable_disable,
        }
        for topic, handler in self.handlers.items():
            bus.on(topic, handler)

    def shutdown(self):
        for topic, handler in self.handlers.items():
            self.bus.remove(topic, handler)

    def add(self, skill_id, intent_name, lang, method, payload):
        with self.lock:
            self.entries[(skill_id, intent_name, lang.lower(), method)] = {"enabled": True, "definition": payload}
        self.on_change()

    def _on_register(self, message):
        data = message.data
        if session_of(message) != DEFAULT_SESSION or not all(data.get(k) for k in ("skill_id", "intent_name", "lang")):
            return
        method = "keyword" if message.msg_type.endswith(".keyword") else "template"
        self.add(data["skill_id"], data["intent_name"], data["lang"], method, data)

    def _on_deregister(self, message):
        skill_id, name = message.data.get("skill_id") or message.context.get("skill_id"), message.data.get("intent_name")
        lang = (message.data.get("lang") or "").lower()
        if session_of(message) != DEFAULT_SESSION or not (skill_id and name):
            return
        self._drop(lambda k: k[0] == skill_id and k[1] == name and (not lang or k[2] == lang))

    def _on_skill_deregister(self, message):
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if session_of(message) == DEFAULT_SESSION and skill_id:
            self._drop(lambda k: k[0] == skill_id)

    def _on_enable_disable(self, message):
        skill_id, name = message.data.get("skill_id") or message.context.get("skill_id"), message.data.get("intent_name")
        if session_of(message) != DEFAULT_SESSION:
            return
        enabled = message.msg_type.endswith(".enable")
        with self.lock:
            hits = [e for k, e in self.entries.items() if k[0] == skill_id and k[1] == name]
            for e in hits:
                e["enabled"] = enabled
        if hits:
            self.on_change()

    def _drop(self, match):
        with self.lock:
            gone = [k for k in self.entries if match(k)]
            for k in gone:
                del self.entries[k]
        if gone:
            self.on_change()

    def sync(self):
        """Pull everything ovos-core's manifest already holds. Call from a worker thread, never a bus handler."""
        from ovos_bus_client.message import Message
        listing = self.bus.wait_for_response(Message("ovos.intent.list", {}), "ovos.intent.list.response",
                                             timeout=SYNC_TIMEOUT_S)
        if not listing or not listing.data.get("ok"):
            return 0
        added = 0
        for skill_id in sorted({i["skill_id"] for i in listing.data.get("intents") or []}):
            reply = self.bus.wait_for_response(Message("ovos.intent.describe", {"skill_id": skill_id}),
                                               "ovos.intent.describe.response", timeout=SYNC_TIMEOUT_S)
            for d in (reply.data.get("definitions") or []) if reply and reply.data.get("ok") else []:
                if d.get("session_id", DEFAULT_SESSION) == DEFAULT_SESSION:
                    with self.lock:
                        key = (d["skill_id"], d["intent_name"], d["lang"].lower(), d["method"])
                        self.entries.setdefault(key, {"enabled": True, "definition": d["definition"]})
                    added += 1
        if added:
            self.on_change()
        return added

    def add_ocp_templates(self, lang="en-us"):
        try:
            import ocp_pipeline
        except ImportError:
            return 0
        base = os.path.join(os.path.dirname(ocp_pipeline.__file__), "locale", lang)
        added = 0
        for fname, name in OCP_INTENTS.items():
            path = os.path.join(base, fname)
            if os.path.exists(path):
                with open(path) as f:
                    samples = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                self.add(OCP_SKILL_IDS.get(name, "ocp"), name, lang, "template", {"samples": samples})
                added += 1
        return added

    def definitions(self):
        """{skill_id: [definition, ...]} in the router's manifest shape; keyword intents become method "adapt"."""
        with self.lock:
            items = sorted((k, dict(e)) for k, e in self.entries.items() if e["enabled"])
        out = {}
        for (skill_id, name, lang, method), e in items:
            payload = e["definition"]
            definition = adapt_definition(payload) if method == "keyword" else payload
            out.setdefault(skill_id, []).append({"skill_id": skill_id, "intent_name": name, "lang": lang,
                                                 "method": "adapt" if method == "keyword" else "template",
                                                 "definition": definition})
        return out
