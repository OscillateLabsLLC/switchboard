from ovos_bus_client.message import Message

from conftest import DATETIME, HA, WEATHER, register_all
from ovos_switchboard.registry import IntentRegistry


def test_registrations_become_router_definitions(bus):
    changes = []
    reg = IntentRegistry(bus, on_change=lambda: changes.append(1))
    register_all(bus)
    defs = reg.definitions()
    assert {d["intent_name"] for d in defs[HA]} == {"turn.on", "turn.off"}
    assert defs[DATETIME][0]["definition"]["samples"]
    assert changes


def test_keyword_registration_becomes_an_adapt_rule(bus):
    reg = IntentRegistry(bus)
    register_all(bus)
    weather = reg.definitions()[WEATHER][0]
    assert weather["method"] == "adapt"
    rule = weather["definition"]
    assert rule["steps"] == [["require", ["Weather"]], ["optionally", ["Location"]]]
    assert rule["vocab"]["location"] == ["in seattle", "in paris", "in london"]
    assert "weather" in rule["keyword_examples"]


def test_deregistration_and_disable_remove_intents(bus):
    reg = IntentRegistry(bus)
    register_all(bus)
    bus.emit(Message("ovos.intent.deregister", {"skill_id": HA, "intent_name": "turn.off"}))
    assert [d["intent_name"] for d in reg.definitions()[HA]] == ["turn.on"]
    bus.emit(Message("ovos.intent.disable", {"skill_id": HA, "intent_name": "turn.on"}))
    assert HA not in reg.definitions()
    bus.emit(Message("ovos.intent.enable", {"skill_id": HA, "intent_name": "turn.on"}))
    assert HA in reg.definitions()
    bus.emit(Message("ovos.skill.deregister", {"skill_id": DATETIME}))
    assert DATETIME not in reg.definitions()


def test_other_sessions_do_not_train_the_router(bus):
    reg = IntentRegistry(bus)
    bus.emit(Message("ovos.intent.register.template",
                     {"skill_id": HA, "intent_name": "turn.on", "lang": "en-us", "samples": ["turn on {entity}"]},
                     {"session": {"session_id": "remote-client"}}))
    assert reg.definitions() == {}


def test_sync_pulls_what_core_already_holds(bus):
    def answer_list(message):
        bus.emit(message.response({"ok": True, "intents": [{"skill_id": HA, "intent_name": "turn.on"}]}))

    def answer_describe(message):
        bus.emit(message.response({"ok": True, "definitions": [
            {"skill_id": HA, "intent_name": "turn.on", "lang": "en-US", "method": "template",
             "session_id": "default", "definition": {"samples": ["turn on {entity}"]}}]}))
    bus.on("ovos.intent.list", answer_list)
    bus.on("ovos.intent.describe", answer_describe)
    reg = IntentRegistry(bus)
    assert reg.sync() == 1
    assert reg.definitions()[HA][0]["lang"] == "en-us"
