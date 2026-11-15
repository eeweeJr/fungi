"""Tests for fungai.eval — metrics, calibration, cost matrix, gallery, report."""

import numpy as np
import pytest

from fungai.eval.calibration import (
    accuracy_at_coverage,
    expected_calibration_error,
    open_set_detection_rate,
)
from fungai.eval.cost_matrix import (
    build_cost_matrix,
    cost_sensitive_weights,
    toxic_edible_misclassifications,
)
from fungai.eval.gallery import build_failure_gallery
from fungai.eval.metrics import (
    genus_confusion_matrix,
    macro_f1,
    rarity_buckets,
    stratified_breakdown,
    top_k_accuracy,
)
from fungai.eval.report import aggregate_by_observation, build_evaluation_report


# --- metrics ---


def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 3, 4])
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_macro_f1_ignores_open_set():
    y_true = np.array([0, 1, -1, -1])
    y_pred = np.array([0, 1, 2, 3])
    assert macro_f1(y_true, y_pred) == pytest.approx(1.0)


def test_top_k_accuracy_k1(dummy_probs, dummy_labels):
    acc = top_k_accuracy(dummy_labels, dummy_probs, k=1)
    assert 0.0 <= acc <= 1.0


def test_top_k_accuracy_k5(dummy_probs, dummy_labels):
    acc5 = top_k_accuracy(dummy_labels, dummy_probs, k=5)
    acc1 = top_k_accuracy(dummy_labels, dummy_probs, k=1)
    assert acc5 >= acc1


# --- calibration ---


def test_ece_perfect_calibration():
    n = 200
    probs = np.zeros((n, 5))
    labels = np.arange(n) % 5
    for i in range(n):
        probs[i, labels[i]] = 1.0
    ece = expected_calibration_error(probs, labels)
    assert ece < 0.05


def test_ece_overconfident():
    n = 100
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=n)
    probs = np.zeros((n, 5))
    probs[np.arange(n), rng.integers(0, 5, size=n)] = 1.0
    ece = expected_calibration_error(probs, labels)
    assert ece > 0.3


def test_accuracy_at_coverage_full(dummy_probs, dummy_labels):
    result = accuracy_at_coverage(dummy_probs, dummy_labels)
    assert result["coverage"][-1] == pytest.approx(1.0, abs=0.05)
    acc_full = result["accuracy"][-1]
    acc_top1 = top_k_accuracy(dummy_labels, dummy_probs, k=1)
    assert abs(acc_full - acc_top1) < 0.05


def test_coverage_monotonic(dummy_probs, dummy_labels):
    result = accuracy_at_coverage(dummy_probs, dummy_labels)
    covs = result["coverage"]
    for i in range(1, len(covs)):
        assert covs[i] >= covs[i - 1]


def test_open_set_detection_no_open():
    probs = np.ones((10, 3)) / 3
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    result = open_set_detection_rate(probs, labels)
    assert result["n_open_set"] == 0
    assert result["false_acceptance_rate"] is None


def test_open_set_detection_high_conf():
    probs = np.zeros((10, 3))
    probs[:, 0] = 0.99
    probs[:, 1] = 0.005
    probs[:, 2] = 0.005
    labels = np.full(10, -1)
    result = open_set_detection_rate(probs, labels, confidence_threshold=0.5)
    assert result["n_open_set"] == 10
    assert result["false_acceptance_rate"] == pytest.approx(1.0)


# --- cost matrix ---


def test_cost_matrix_shape(dummy_species_map, dummy_toxicity_map):
    cm = build_cost_matrix(dummy_species_map, dummy_toxicity_map)
    n = len(dummy_species_map)
    assert cm.shape == (n, n)
    assert np.all(np.diag(cm) == 0.0)


def test_cost_sensitive_weights_shape(dummy_species_map, dummy_toxicity_map):
    w = cost_sensitive_weights(dummy_species_map, dummy_toxicity_map)
    assert w.shape == (len(dummy_species_map),)
    assert w[0] > 1.0  # species_A is toxic


def test_toxic_edible_tracking(dummy_species_map, dummy_toxicity_map):
    # species_A(0) is toxic, species_B(1) is edible
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])  # first sample: toxic predicted as edible
    result = toxic_edible_misclassifications(y_true, y_pred, dummy_species_map, dummy_toxicity_map)
    assert result["toxic_predicted_edible_count"] == 1
    assert result["edible_predicted_toxic_count"] == 1


# --- genus confusion matrix ---


def test_genus_confusion_matrix_shape():
    species_to_genus = {0: "A", 1: "A", 2: "B", 3: "B", 4: "C", 5: "C"}
    genus_names = ["A", "B", "C", "D", "E", "F"]
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 2, 2, 4, 4, 5])
    cm = genus_confusion_matrix(y_true, y_pred, species_to_genus, genus_names)
    assert cm.shape == (6, 6)


# --- rarity ---


def test_rarity_buckets():
    counts = {0: 5, 1: 25, 2: 100, 3: 500}
    b = rarity_buckets(counts)
    assert b[0] == "rare"
    assert b[1] == "few"
    assert b[2] == "medium"
    assert b[3] == "common"


# --- gallery ---


def test_failure_gallery_selects_worst():
    probs = np.array([
        [0.1, 0.9],  # pred=1, true=0 → wrong, conf=0.9
        [0.8, 0.2],  # pred=0, true=0 → correct
        [0.3, 0.7],  # pred=1, true=0 → wrong, conf=0.7
    ])
    y_true = np.array([0, 0, 0])
    gallery = build_failure_gallery(probs, y_true, ["/a", "/b", "/c"], ["sp0", "sp1"], n=5)
    assert len(gallery) == 2
    assert gallery[0]["confidence"] > gallery[1]["confidence"]
    assert gallery[0]["rank"] == 1


# --- report ---


def test_build_evaluation_report_keys(dummy_probs, dummy_labels, dummy_species_map, dummy_toxicity_map):
    species_to_genus = {i: f"genus_{i % 3}" for i in range(5)}
    genus_names = ["genus_0", "genus_1", "genus_2"]
    train_counts = {i: 50 for i in range(5)}
    predictions = {
        "probs": dummy_probs,
        "labels": dummy_labels,
        "image_paths": [f"/img_{i}.jpg" for i in range(100)],
        "observation_ids": [f"obs_{i}" for i in range(100)],
    }
    report = build_evaluation_report(
        predictions, dummy_species_map, dummy_toxicity_map,
        species_to_genus, genus_names, train_counts,
    )
    required = [
        "timestamp", "n_samples", "n_observations", "n_open_set",
        "primary_metrics", "calibration", "coverage", "open_set",
        "cost_sensitive", "genus_confusion_matrix", "stratified", "failure_gallery",
    ]
    for k in required:
        assert k in report, f"Missing key: {k}"
    assert "macro_f1" in report["primary_metrics"]
    assert "by_genus" in report["stratified"]


def test_aggregate_by_observation():
    probs = np.array([
        [0.8, 0.2],
        [0.6, 0.4],
        [0.1, 0.9],
    ])
    labels = np.array([0, 0, 1])
    obs_ids = ["obs_1", "obs_1", "obs_2"]
    agg_probs, agg_labels = aggregate_by_observation(probs, labels, obs_ids)
    assert agg_probs.shape[0] == 2
    obs1_row = agg_probs[agg_labels == 0][0]
    assert obs1_row[0] == pytest.approx(0.7)
    assert obs1_row[1] == pytest.approx(0.3)


# --- genus and open-set ---


def test_genus_probs_sums_species_within_genus():
    from fungai.eval.metrics import genus_probs

    probs = np.array([[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]])
    out = genus_probs(probs, {0: "A", 1: "A", 2: "B"}, ["A", "B"])
    assert out == pytest.approx(np.array([[0.3, 0.7], [0.9, 0.1]]))


def test_genus_accuracy_macro_differs_from_plain():
    from fungai.eval.metrics import genus_accuracy

    # genus 0: 8 samples all right; genus 1: 2 samples both wrong
    g_true = np.array([0] * 8 + [1] * 2)
    g_probs = np.tile([0.9, 0.1], (10, 1))
    out = genus_accuracy(g_true, g_probs, ["A", "B"])
    assert out["accuracy"] == pytest.approx(0.8)
    assert out["macro_accuracy"] == pytest.approx(0.5)
    assert out["per_genus_recall"] == {"A": 1.0, "B": 0.0}
    assert genus_accuracy(np.array([-1]), g_probs[:1], ["A", "B"])["n_samples"] == 0


def test_error_decomposition_sums_to_one():
    from fungai.eval.metrics import error_decomposition

    s2g = {0: "A", 1: "A", 2: "B"}
    y_true = np.array([0, 0, 0, 2, -1])
    y_pred = np.array([0, 1, 2, 2, 0])
    out = error_decomposition(y_true, y_pred, s2g)
    assert out["correct"] == pytest.approx(0.5)
    assert out["right_genus_wrong_species"] == pytest.approx(0.25)
    assert out["wrong_genus"] == pytest.approx(0.25)
    assert out["n_samples"] == 4


def test_open_set_metrics_separable_and_fixed_threshold():
    from fungai.eval.calibration import open_set_metrics

    known = np.tile([0.9, 0.1], (20, 1))
    unknown = np.tile([0.55, 0.45], (10, 1))
    probs = np.vstack([known, unknown])
    y = np.array([0] * 20 + [-1] * 10)
    out = open_set_metrics(probs, y)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["unknown_false_acceptance_rate"] == 0.0
    assert out["known_retained"] == pytest.approx(1.0)
    loose = open_set_metrics(probs, y, threshold=0.5)
    assert loose["unknown_false_acceptance_rate"] == 1.0
    assert open_set_metrics(known, np.zeros(20, dtype=int))["auroc"] is None


def test_report_genus_site_and_gallery_use_observation_level(dummy_species_map, dummy_toxicity_map):
    # 3 observations x 2 images; the last observation is an unknown species of genus 1
    probs = np.array([
        [0.9, 0.1, 0, 0, 0], [0.7, 0.3, 0, 0, 0],      # obs a: species 0, right
        [0.2, 0.8, 0, 0, 0], [0.4, 0.6, 0, 0, 0],      # obs b: species 0, predicted 1 (same genus)
        [0, 0, 0.6, 0.4, 0], [0, 0, 0.5, 0.5, 0],      # obs c: unknown, genus 1
    ])
    predictions = {
        "probs": probs,
        "labels": np.array([0, 0, 0, 0, -1, -1]),
        "image_paths": [f"/img_{i}.jpg" for i in range(6)],
        "observation_ids": ["a", "a", "b", "b", "c", "c"],
        "genus_labels": np.array([0, 0, 0, 0, 1, 1]),
        "revisited": np.array([True, True, False, False, False, False]),
    }
    s2g = {0: "g0", 1: "g0", 2: "g1", 3: "g1", 4: "g1"}
    report = build_evaluation_report(
        predictions, dummy_species_map, dummy_toxicity_map, s2g, ["g0", "g1"], {i: 50 for i in range(5)},
    )
    assert report["n_open_set_observations"] == 1
    assert report["genus"]["known"]["accuracy"] == 1.0
    assert report["genus"]["unknown"] == {
        "accuracy": 1.0, "macro_accuracy": 1.0, "per_genus_recall": {"g1": 1.0}, "n_samples": 1,
    }
    assert report["error_decomposition"]["right_genus_wrong_species"] == pytest.approx(0.5)
    assert report["stratified"]["by_site"]["revisited"]["n_samples"] == 1
    assert report["stratified"]["by_site"]["revisited"]["top_1_accuracy"] == 1.0
    assert report["stratified"]["by_site"]["new_site"]["top_1_accuracy"] == 0.0
    assert report["stratified"]["by_toxicity"]["edible"]["n_samples"] == 0  # unknowns not counted as edible
    assert report["open_set"]["auroc"] is not None
    assert [g["image_path"] for g in report["failure_gallery"]] == ["/img_2.jpg"]
