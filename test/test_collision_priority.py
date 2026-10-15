import pytest

from ovos_switchboard.core.device import Device
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.options import ExactOptions, options_from
from ovos_switchboard.core.schema import Schema


ALERTS, HA = "ovos-skill-alerts.openvoiceos", "skill-homeassistant.oscillatelabsllc"


def template(intent_name, *samples):
    return {"intent_name": intent_name, "lang": "en-us", "method": "template",
            "definition": {"samples": list(samples)}}


@pytest.fixture(name="device", scope="module")
def device_fixture():
    # the collision measured on a real install: alerts' "turn off {alertkind}" vs Home Assistant's "turn off {entity}"
    return Device.from_definitions({ALERTS: [template("cancel_alert", "turn off (the|my|) {alertkind}")],
                                    HA: [template("turn.off", "turn off (the|my|) {entity}")]})


def owner(device, **options):
    return ExactLayer(device, Schema.load(), options=ExactOptions(**options)).route("turn off the living room lights")


def test_a_tied_collision_goes_to_the_classifier_by_default(device):
    assert owner(device) is None


@pytest.mark.parametrize("priority, intent", [("user", "device.off"), ("core", "timer.cancel")])
def test_collision_priority_picks_the_one_owner_in_its_group(device, priority, intent):
    assert owner(device, collision_priority=priority) == intent


def test_no_owner_wins_when_the_group_has_two(device):
    assert owner(device, collision_priority="user", core_skills=()) is None  # both count as user


@pytest.mark.parametrize("skill_priority, intent", [((HA,), "device.off"), ((ALERTS, HA), "timer.cancel"),
                                                    ((".oscillatelabsllc",), "device.off")])
def test_the_earliest_listed_skill_wins(device, skill_priority, intent):
    assert owner(device, skill_priority=skill_priority) == intent


def test_the_list_overrides_the_group(device):
    assert owner(device, skill_priority=(HA,), collision_priority="core") == "device.off"


def test_an_unlisted_collision_falls_back_to_the_group(device):
    assert owner(device, skill_priority=("skill-other.example",), collision_priority="core") == "timer.cancel"
    assert owner(device, skill_priority=("skill-other.example",)) is None


def test_skill_priority_reads_from_config():
    options = options_from(ExactOptions, {"skill_priority": ["ocp", HA]})  # mycroft.conf gives a list
    assert options.priority_rank(HA) == 1 and options.priority_rank(ALERTS) is None


def test_core_skills_take_suffixes_and_exact_ids():
    options = ExactOptions(core_skills=(".openvoiceos", "ocp"))
    assert options.is_core(ALERTS) and options.is_core("ocp") and not options.is_core(HA)


def test_invalid_priority_is_rejected():
    with pytest.raises(ValueError):
        ExactOptions(collision_priority="mine")


SIBLINGS = "ovos-skill-alerts.openvoiceos"


@pytest.fixture(name="siblings", scope="module")
def siblings_fixture():
    # sibling intents of one skill whose templates overlap, as in the OVOS golden utterances
    return Device.from_definitions({SIBLINGS: [template("change_repeat", "change my {kind} to repeat {when}"),
                                               template("reschedule_alert", "change my {kind} to {time}")]})


def sibling_owner(device, utterance, **options):
    return ExactLayer(device, Schema.load(), options=ExactOptions(**options)).route(utterance)


def test_classifier_mode_sends_every_collision_to_the_classifier(siblings):
    assert sibling_owner(siblings, "change my alarm to repeat every monday", collision_resolution="classifier") is None


def test_by_default_the_template_with_more_literal_words_wins(siblings):
    winner = sibling_owner(siblings, "change my alarm to repeat every monday")
    assert winner.endswith(f"{SIBLINGS}:change_repeat")  # unmapped intents own themselves (__unmapped__:<label>)


def test_specific_counts_the_words_each_match_covers_not_the_template_text():
    # the collision measured in the harness: volume's optional [the] and [to] are matched here, so volume covers
    # "set the volume to" literally where alerts' template covers "set the ... to" and leaves "volume" to a slot
    volume = Device.from_definitions({
        ALERTS: [template("change_media_properties", "(change|set) (my|the) {schedkind} to {mediaform}")],
        "ovos-skill-volume.openvoiceos": [template("change_volume", "(change|set|make) [the] volume [to] {amount}")]})
    layer = ExactLayer(volume, Schema.load(), options=ExactOptions(collision_resolution="specific"))
    assert layer.route("set the volume to five") == "volume.set"


def test_specific_leaves_a_tie_to_the_classifier(device):
    assert owner(device, collision_resolution="specific") is None  # alerts' and HA's "turn off the {x}": a tie


def test_priority_lists_decide_before_specificity(device):
    assert owner(device, collision_resolution="specific", skill_priority=(HA,)) == "device.off"


def test_invalid_collision_resolution_is_rejected():
    with pytest.raises(ValueError):
        ExactOptions(collision_resolution="guess")
