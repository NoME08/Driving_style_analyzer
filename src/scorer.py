"""Driving score — percentile-based multi-dimension scoring against personal history.

Each trip receives four scores (0-100, higher = better):
  - safety      : hard braking, hard acceleration, acceleration volatility
  - smoothness  : jerk, speed variation, acceleration magnitude
  - efficiency  : stop time, cruise ratio, RPM/throttle stability
  - composite   : weighted average of the three
"""

import numpy as np


# ── Dimension definitions ──────────────────────────────
# Each: (feature_key, direction) where direction = -1 means lower-is-better

SAFETY_METRICS = [
    ("pct_hard_brake",  -1),
    ("pct_hard_accel",  -1),
    ("acc_std",         -1),
]

SMOOTHNESS_METRICS = [
    ("acc_std",         -1),
    ("acc_mag_mean",    -1),
    ("speed_std",       -1),
]

EFFICIENCY_METRICS = [
    ("stop_pct",        -1),
    ("cruise_pct",       1),   # higher = more cruise = better
    ("rpm_std",         -1),
    ("throttle_std",    -1),
]

# Composite weights
WEIGHTS = {
    "safety":      0.35,
    "smoothness":  0.35,
    "efficiency":  0.30,
}


def _percentile_rank(values, target, direction):
    """Compute percentile rank of target within values.

    direction =  1 → higher-is-better (rank from top)
    direction = -1 → lower-is-better (inverted rank)
    """
    values = np.array(values)
    if len(values) < 1:
        return 50.0  # no history

    count_le = np.sum(values <= target)
    pct = (count_le / len(values)) * 100

    if direction == -1:
        # Invert: lower values get higher percentiles
        pct = 100 - pct

    return round(float(pct), 1)


def score_trip(features, history_features):
    """Score a single trip against a history of feature dicts.

    Parameters
    ----------
    features : dict
        Single trip's feature dict from extract_features().
    history_features : list[dict]
        All historical feature dicts (including this trip).

    Returns
    -------
    dict with keys: safety, smoothness, efficiency, composite
    """
    if not history_features:
        history_features = [features]

    scores = {}

    for dim_name, metrics in [
        ("safety",      SAFETY_METRICS),
        ("smoothness",  SMOOTHNESS_METRICS),
        ("efficiency",  EFFICIENCY_METRICS),
    ]:
        dim_scores = []
        for feat_key, direction in metrics:
            val = features.get(feat_key)
            if val is None:
                continue
            hist_vals = []
            for hf in history_features:
                hv = hf.get(feat_key)
                if hv is not None:
                    hist_vals.append(hv)

            if hist_vals:
                dim_scores.append(_percentile_rank(hist_vals, val, direction))

        if dim_scores:
            scores[dim_name] = round(float(np.mean(dim_scores)), 1)
        else:
            scores[dim_name] = 50.0

    # Composite
    composite = 0.0
    for dim, weight in WEIGHTS.items():
        composite += scores.get(dim, 50) * weight
    scores["composite"] = round(composite, 1)

    return scores


def score_all(features_list):
    """Score all trips in a list, each against the full history.

    Returns list of score dicts in the same order.
    """
    results = []
    for feat in features_list:
        results.append(score_trip(feat, features_list))
    return results


def score_label(score):
    """Human-readable label for a composite score."""
    if score >= 85:
        return "卓越", "#34c759"
    elif score >= 70:
        return "优秀", "#007aff"
    elif score >= 55:
        return "良好", "#ff9500"
    elif score >= 40:
        return "一般", "#ff3b30"
    else:
        return "待改善", "#ff3b30"



