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
