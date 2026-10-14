import pytest

from ovos_switchboard.core.device import Device
from ovos_switchboard.core.exact import ExactLayer
from ovos_switchboard.core.options import ExactOptions
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


def test_a_collision_goes_to_the_classifier_by_default(device):
    assert owner(device) is None


@pytest.mark.parametrize("priority, intent", [("user", "device.off"), ("core", "timer.cancel")])
def test_collision_priority_picks_the_one_owner_in_its_group(device, priority, intent):
    assert owner(device, collision_priority=priority) == intent


def test_no_owner_wins_when_the_group_has_two(device):
    assert owner(device, collision_priority="user", core_skills=()) is None  # both count as user


def test_core_skills_take_suffixes_and_exact_ids():
    options = ExactOptions(core_skills=(".openvoiceos", "ocp"))
    assert options.is_core(ALERTS) and options.is_core("ocp") and not options.is_core(HA)


def test_invalid_priority_is_rejected():
    with pytest.raises(ValueError):
        ExactOptions(collision_priority="mine")
