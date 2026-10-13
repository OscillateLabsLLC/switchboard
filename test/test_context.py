from ovos_switchboard.context import SpeechContext


def test_unknown_until_the_assistant_speaks():
    assert SpeechContext().features("turn on the lamp") == {"gap_s": None, "echo_run": 0}


def test_gap_and_echo_run():
    ctx = SpeechContext(clock=lambda: 1000.0)
    ctx.heard_assistant("The living room lamp is now on.", ts=990.0)
    f = ctx.features("the living room lamp is now", now=1000.0)
    assert f == {"gap_s": 10.0, "echo_run": 6}
    # a request reuses WORDS, not the phrase
    assert ctx.features("turn off the lamp", now=1000.0)["echo_run"] <= 2


def test_echo_window_and_sessions():
    ctx = SpeechContext()
    ctx.heard_assistant("it is going to rain later today", session_id="a", ts=0.0)
    assert ctx.features("it is going to rain later today", session_id="a", now=61.0)["echo_run"] == 0
    assert ctx.features("it is going to rain", session_id="a", now=30.0)["echo_run"] == 5
    assert ctx.features("it is going to rain", session_id="b", now=30.0)["gap_s"] is None
