"""Tests for baseline probability generation logic."""

import numpy as np
import pytest


def test_random_baseline_probs_sum_to_one():
    rng = np.random.default_rng(42)
    num_classes = 5
    n = 50
    probs = rng.dirichlet(np.ones(num_classes), size=n)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_majority_baseline_all_same():
    num_classes = 5
    n = 50
    train_counts = {0: 10, 1: 100, 2: 5, 3: 20, 4: 8}
    majority_class = max(train_counts, key=train_counts.get)
    probs = np.zeros((n, num_classes))
    probs[:, majority_class] = 1.0
    preds = probs.argmax(axis=1)
    assert np.all(preds == majority_class)
    assert majority_class == 1


def test_class_prior_matches_distribution():
    num_classes = 5
    n = 10000
    train_counts = {0: 100, 1: 200, 2: 50, 3: 150, 4: 500}
    total = sum(train_counts.values())
    prior = np.zeros(num_classes)
    for cls, count in train_counts.items():
        prior[cls] = count / total
    prior = prior / prior.sum()
    probs = np.tile(prior, (n, 1))

    assert probs.shape == (n, num_classes)
    np.testing.assert_allclose(probs[0], prior)
    assert probs[0, 4] == pytest.approx(500 / total)
