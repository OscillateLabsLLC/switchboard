import numpy as np

from ovos_switchboard.adapters.isolated_training import train_isolated
from ovos_switchboard.core.classifier import Classifier, training_data
from test_stages import Parts


def test_child_process_training_gives_the_in_process_model_exactly():
    parts = Parts()
    rows = training_data(parts.device, parts.schema, parts.synthetic)
    (want, want_meta), (got, got_meta) = Classifier.train(rows).state(), train_isolated(rows).state()
    assert got_meta == want_meta
    assert want.keys() == got.keys() and all(np.array_equal(want[k], got[k]) for k in want)
