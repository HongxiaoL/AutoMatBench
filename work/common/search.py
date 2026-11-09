"""Finite categorical RF/EI search and the discrete LHS baseline."""

import random

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from . import params


def candidates(min_neighbors=1, exclude_reference=False):
    return [
        (r, n, s)
        for r in params.R_values
        for n in params.N_values
        for s in params.S_values
        if s >= min_neighbors
        and not (n == 60 and s >= 9)
        and not (exclude_reference and (r, n, s) == (0.0, 48, 10))
    ]


def one_hot(config):
    return np.array(
        [float(value == level) for value, levels in zip(
            config, (params.R_values, params.N_values, params.S_values)
        ) for level in levels]
    )


def expected_improvement(mu, sigma, best):
    sigma = np.maximum(sigma, 1e-9)
    difference = mu - best
    z = difference / sigma
    return difference * norm.cdf(z) + sigma * norm.pdf(z)


def select_ei(grid, observations, values, rng, trees=100):
    """Values are expressed in maximization form, not necessarily raw MAE."""
    visited = set(observations)
    remaining = [c for c in grid if c not in visited]
    if not remaining:
        raise ValueError("No unvisited configurations remain.")
    if not values:
        return rng.choice(remaining)
    forest = RandomForestRegressor(
        n_estimators=trees, max_depth=None, random_state=42, n_jobs=1
    )
    forest.fit(np.vstack([one_hot(c) for c in observations]), values)
    # Preserve float32 ensemble arithmetic and full-grid prediction.
    encoded = np.vstack([one_hot(c) for c in grid])
    predictions = np.array(
        [tree.predict(encoded) for tree in forest.estimators_], dtype=np.float32
    )
    indices = [i for i, c in enumerate(grid) if c not in visited]
    mu = predictions.mean(axis=0)[indices]
    sigma = np.sqrt(np.maximum(predictions.var(axis=0)[indices], 1e-9))
    ei = expected_improvement(mu, sigma, max(values))
    position = int(np.argmax(ei))
    return remaining[position] if ei[position] > 0 else rng.choice(remaining)


def sample_lhs(grid, budget, rng):
    """Discrete LHS baseline using axis permutations with cell replacement; not scipy.qmc.LatinHypercube."""
    sequences = []
    for levels in (params.R_values, params.N_values, params.S_values):
        base = list(levels)
        rng.shuffle(base)
        sequence = [base[i % len(base)] for i in range(budget)]
        sequences.append(sequence)
    for sequence in sequences:
        rng.shuffle(sequence)
    valid = set(grid)
    picks = []
    seen = set()
    for config in zip(*sequences):
        if config not in valid or config in seen:
            config = rng.choice([c for c in grid if c not in seen])
        picks.append(config)
        seen.add(config)
    return picks


def replay(truth, budget=24, seed=42, method="bo", n_init=None, trees=100):
    grid = [c for c in candidates() if c in truth and np.isfinite(truth[c])]
    if not 1 <= budget <= len(grid):
        raise ValueError("Budget must lie between 1 and the available grid size.")
    rng = random.Random(seed)
    if method == "rs":
        picks = rng.sample(grid, budget)
    elif method == "lhs":
        picks = sample_lhs(grid, budget, rng)
    elif method == "bo":
        initial = max(1, round(0.3 * budget)) if n_init is None else n_init
        if not 1 <= initial <= budget:
            raise ValueError("Initial sample size must lie between 1 and budget.")
        picks = rng.sample(grid, initial)
        while len(picks) < budget:
            scores = [-float(truth[c]) for c in picks]
            picks.append(select_ei(grid, picks, scores, rng, trees=trees))
    else:
        raise ValueError(f"Unknown search method: {method}")
    return picks, [float(truth[c]) for c in picks]
