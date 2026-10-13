import os
import pathlib

import numpy as np

from conftest import DIALOG_LINES, register_all
from ovos_switchboard.adapters.model_cache import ModelCache
from ovos_switchboard.ovos.runtime import Settings, SwitchboardRuntime
from ovos_switchboard.ovos.steps import ClassifierStep, ExactStep


class Builds:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return {"model": self.count}


class DictCodec:
    @staticmethod
    def state(model):
        return {"weights": np.array([model["model"]])}, {"model": model["model"]}

    @staticmethod
    def from_state(arrays, meta):
        assert arrays["weights"][0] == meta["model"]
        return {"model": meta["model"]}


class Payload:
    """Unpickling this creates `marker`: proof that the pickle ran."""
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return pathlib.Path.touch, (self.marker,)


def test_trains_once_then_loads(tmp_path):
    cache, build = ModelCache(str(tmp_path)), Builds()
    assert cache.get_or_build("classifier", ["rows"], build, DictCodec) == ({"model": 1}, False)
    assert cache.get_or_build("classifier", ["rows"], build, DictCodec) == ({"model": 1}, True)
    assert cache.get_or_build("classifier", ["other rows"], build, DictCodec) == ({"model": 2}, False)
    assert build.count == 2


def test_an_unreadable_entry_is_retrained(tmp_path):
    cache, build = ModelCache(str(tmp_path)), Builds()
    cache.get_or_build("classifier", ["rows"], build, DictCodec)
    (entry,) = tmp_path.glob("classifier-*.npz")
    entry.write_bytes(b"not a model")
    assert cache.get_or_build("classifier", ["rows"], build, DictCodec) == ({"model": 2}, False)


def test_a_pickle_in_the_cache_is_never_executed(tmp_path):
    cache, build = ModelCache(str(tmp_path)), Builds()
    cache.get_or_build("classifier", ["rows"], build, DictCodec)
    (entry,) = tmp_path.glob("classifier-*.npz")
    marker = tmp_path / "payload-ran"
    payload = np.empty(1, dtype=object)
    payload[0] = Payload(marker)
    with open(entry, "wb") as f:
        np.savez(f, __meta__=np.array('{"model": 1}'), weights=payload)
    assert cache.get_or_build("classifier", ["rows"], build, DictCodec) == ({"model": 2}, False)
    assert not marker.exists()


def test_only_the_newest_entries_are_kept(tmp_path):
    cache, build = ModelCache(str(tmp_path), max_entries=2), Builds()
    for i in range(4):
        cache.get_or_build("classifier", [i], build, DictCodec)
        os.utime(max(tmp_path.glob("classifier-*.npz"), key=os.path.getmtime), (i, i))
    assert len(list(tmp_path.glob("classifier-*.npz"))) == 2


def test_disabled_cache_always_builds(tmp_path):
    cache, build = ModelCache(str(tmp_path), enabled=False), Builds()
    cache.get_or_build("classifier", ["rows"], build, DictCodec)
    cache.get_or_build("classifier", ["rows"], build, DictCodec)
    assert build.count == 2 and not list(tmp_path.iterdir())


def test_classifier_step_reuses_its_model_across_restarts(bus, tmp_path):
    def runtime():
        return SwitchboardRuntime(bus, Settings(retrain_debounce_s=3600, decision_log=False,
                                                model_cache_path=str(tmp_path)),
                                  dialog_lines=lambda lang: DIALOG_LINES, start=False)
    first = runtime()
    ClassifierStep(bus, {}, runtime=first)
    register_all(bus)
    first.rebuild()
    first.shutdown()
    second = runtime()
    ClassifierStep(bus, {}, runtime=second)
    register_all(bus)
    hits = []
    get_or_build = second.models.get_or_build
    def recording(*args):
        model, hit = get_or_build(*args)
        hits.append(hit)
        return model, hit
    second.models.get_or_build = recording
    second.rebuild()
    assert hits == [True]
    assert second.stage("classifier").decide("whats the time right now", {}).intent == "time.current"


def test_slow_stages_build_after_the_others(bus):
    rt = SwitchboardRuntime(bus, Settings(retrain_debounce_s=3600, decision_log=False), start=False)
    order = []
    rt.provide("classifier", lambda b: order.append("classifier"), slow=True)
    ExactStep(bus, {}, runtime=rt)
    rt.builders["exact"] = (lambda b, f=rt.builders["exact"][0]: order.append("exact") or f(b), False)
    register_all(bus)
    rt.rebuild()
    assert order == ["exact", "classifier"]
