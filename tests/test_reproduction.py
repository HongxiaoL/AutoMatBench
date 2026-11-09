import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work"))

from audit import fold_shift, source_fraction
from common.results import load_state
from common.search import candidates, expected_improvement, one_hot, replay, sample_lhs
from optimize import metric


def test_grid_constraints_and_reference():
    grid = candidates()
    assert len(grid) == len(set(grid)) == 858
    assert len(candidates(exclude_reference=True)) == 857
    assert len(candidates(min_neighbors=4)) == 594
    assert len(candidates(min_neighbors=6)) == 418
    assert (0.0, 48, 10) not in candidates(exclude_reference=True)
    assert all(not (n == 60 and s >= 9) for _, n, s in grid)


def test_one_hot():
    for config in candidates():
        vector = one_hot(config)
        assert vector.shape == (29,)
        assert vector.sum() == 3


@pytest.mark.parametrize("method", ["bo", "rs", "lhs"])
def test_replay_is_repeatable_and_without_replacement(method):
    truth = {c: (c[0] - 0.2) ** 2 + c[1] * 0.01 + c[2] * 0.001
             for c in candidates()}
    first = replay(truth, 12, 42, method)
    assert first == replay(truth, 12, 42, method)
    picks, ys = first
    assert len(picks) == len(set(picks)) == len(ys) == 12
    assert np.all(np.diff(np.minimum.accumulate(ys)) <= 0)


def test_lhs_rejects_invalid_or_repeated_cells():
    for seed in range(10):
        picks = sample_lhs(candidates(), 36, random.Random(seed))
        assert len(set(picks)) == 36
        assert set(picks) <= set(candidates())


def test_invalid_budget_fails():
    with pytest.raises(ValueError):
        replay({c: 1 for c in candidates()}, 859)
    with pytest.raises(ValueError):
        replay({c: 1 for c in candidates()}, 12, n_init=13)


def test_ei_prefers_larger_predicted_objective():
    ei = expected_improvement(np.array([0.0, 2.0]), np.ones(2), 1.0)
    assert ei[1] > ei[0] >= 0


def test_actual_test_members_not_train_complement():
    train = ["mp-1", "mp-2", "JVASP-1"]
    test = ["JVASP-2"]
    assert fold_shift(train, test) == pytest.approx(2 / 3)
    assert source_fraction(["mp-1", "mvc-2", "JVASP-1"]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        fold_shift(train, ["mp-1"])
    with pytest.raises(ValueError):
        source_fraction(["unknown-1"])


def test_state_order_is_not_rebuilt_from_unordered_indices(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "evaluated_idx": [500, 1], "params_obs": [[0.8, 10, 1], [0.1, 10, 7]],
        "y_obs": [0.8, 0.2],
    }))
    configs, values = load_state(path)
    assert configs[0] == (0.8, 10, 1)
    assert values == [0.8, 0.2]


def test_metallicity_score_convention():
    assert metric({"average_metrics": {"mae": 0.125}}, "metallicity") == 0.875
    assert metric({"average_metrics": {"roc_auc": 0.8}}, "metallicity") == 0.8


def test_lightweight_analysis_does_not_import_model_frameworks():
    assert importlib.util.find_spec("common.search") is not None
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules
