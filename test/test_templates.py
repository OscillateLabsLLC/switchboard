import pytest

from ovos_switchboard.core.templates import template_regex


@pytest.mark.parametrize("line, utterance", [
    ("play {query}", "play jazz"),
    ("stop [the] {entity}", "stop the fan"),
    ("stop [the] {entity}", "stop fan"),
    ("hello [world] [please]", "hello"),
    ("hello [world] [please]", "hello please"),
    ("hello [world] [please]", "hello world please"),
    ("what time is it", "what time is it?"),
])
def test_template_matches(line, utterance):
    assert template_regex(line).match(utterance)


# Found by the intent-eval exact audit: the assistant's own "playing white noise" and "starting a speed test" matched
# play/start templates, because a template space could match zero characters inside a word.
@pytest.mark.parametrize("line, utterance", [
    ("play {noise} noise", "playing white noise"),
    ("play {query}", "playing by"),
    ("(play|start) {query}", "starting a speed test"),
    ("call {contact}", "caller id please"),
])
def test_template_words_end_at_word_boundaries(line, utterance):
    assert not template_regex(line).match(utterance)


def test_a_malformed_template_cannot_stop_the_exact_layer_building():
    """gl-ES ovos-skill-news ships lines like "quero escoitar as (noticias|novas" (unclosed group)."""
    from ovos_switchboard.core.device import Device
    from ovos_switchboard.core.exact import ExactLayer
    from ovos_switchboard.core.schema import Schema

    defs = {"news.test": [{"skill_id": "news.test", "intent_name": "news", "lang": "en-us", "method": "template",
                           "definition": {"samples": ["play the (news|headlines", "read me the news",
                                                      "read me {number:count} headlines"]}}]}
    device = Device.from_definitions(defs, "en-us")
    layer = ExactLayer(device, Schema.load())
    assert layer.route("read me the news") == "__unmapped__:news.test:news"
    # a typed slot ({number:count}) is not malformed: it still claims its phrases
    assert layer.route("read me five headlines") == "__unmapped__:news.test:news"
    assert [line for _, line in device.template_entries()] == ["read me the news", "read me {number:count} headlines"]
    assert all(not e["example"].startswith("play the") for e in device.examples), "no rows from the broken line"
