"""Driving style classifier — unsupervised clustering + PCA visualization."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ── Feature subsets for clustering ──────────────────────
# Core features (always available from OBD speed data)
CORE_FEATURES = [
    "speed_mean",        # 均速
    "speed_std",         # 速度波动
    "pct_high_speed",    # 高速占比
    "acc_std",           # 加速度波动
    "acc_mag_mean",      # 加速度幅值
    "pct_hard_accel",    # 急加速%
    "pct_hard_brake",    # 急减速%
    "stop_pct",          # 停止占比
]

# Extended features (OBD PID data, may be missing)
EXTENDED_FEATURES = [
    "rpm_mean",
    "rpm_std",
    "throttle_mean",
    "throttle_std",
]

# Human-readable names
FEATURE_NAMES = {
    "speed_mean": "均速",
    "speed_std": "速度波动",
    "pct_high_speed": "高速占比",
    "acc_std": "加速波动",
    "acc_mag_mean": "加速幅值",
    "pct_hard_accel": "急加速%",
    "pct_hard_brake": "急减速%",
    "stop_pct": "停止占比",
    "rpm_mean": "转速均值",
    "rpm_std": "转速波动",
    "throttle_mean": "节气门均值",
    "throttle_std": "节气门波动",
}

# Cluster profile templates for auto-labeling
STYLE_PROFILES = {
    "激进型":   {"speed_mean": 1, "pct_high_speed": 1, "pct_hard_accel": 1, "acc_std": 1, "stop_pct": -1},
    "平稳型":   {"speed_mean": 0, "acc_std": -1, "acc_mag_mean": -1, "pct_hard_accel": -1, "pct_hard_brake": -1},
    "拥堵通勤": {"stop_pct": 1, "speed_mean": -1, "pct_high_speed": -1, "speed_std": -1},
    "高速巡航": {"pct_high_speed": 1, "speed_mean": 1, "stop_pct": -1, "acc_std": -1},
}


def prepare_features(features_list):
    """Convert list of feature dicts to a numeric matrix.

    Uses core features plus any extended features (RPM, throttle)
    present in ALL trips. Returns (X, feature_names, valid).
    """
    available_ext = list(EXTENDED_FEATURES)
    for feat in features_list:
        for ef in list(available_ext):
            if feat.get(ef) is None and ef in available_ext:
                available_ext.remove(ef)

    use_features = CORE_FEATURES + available_ext
    X = np.zeros((len(features_list), len(use_features)))
    for i, feat in enumerate(features_list):
        for j, f in enumerate(use_features):
            X[i, j] = float(feat.get(f, 0) or 0)

    return X, use_features, [True] * len(features_list)


def find_optimal_k(X, max_k=5):
    """Find optimal K using silhouette score.

    Returns (best_k, scores_dict).
    """
    n = len(X)
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 2, {}

    scores = {}
    best_k, best_score = 2, -1

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = round(score, 3)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores


def cluster_trips(features_list, n_clusters=None):
    """Run K-Means clustering on trip features.

    Parameters
    ----------
    features_list : list[dict]
        List of feature dicts from extract_features()
    n_clusters : int or None
        If None, auto-detect optimal K (2-4)

    Returns
    -------
    dict with keys:
        labels          : cluster label per trip
        centroids       : cluster centroid vectors
        pca_xy          : 2D PCA coordinates for plotting
        feature_names   : features used for clustering
        cluster_profiles: auto-generated labels per cluster
        k_scores        : silhouette scores per K
        scaler, kmeans, pca : fitted model objects
    """
    X, feature_names, valid = prepare_features(features_list)

    # Filter to valid rows only
    valid_idx = [i for i, v in enumerate(valid) if v]
    if len(valid_idx) < 2:
        return None

    X_valid = X[valid_idx]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Find optimal K
    if n_clusters is None:
        n_clusters, k_scores = find_optimal_k(X_scaled)
    else:
        _, k_scores = find_optimal_k(X_scaled, max_k=5)

    n_clusters = max(2, min(n_clusters, len(X_valid) - 1))

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_full = np.full(len(features_list), -1, dtype=int)
    labels_full[valid_idx] = kmeans.fit_predict(X_scaled)

    # Centroids in original scale
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_xy = np.full((len(features_list), 2), np.nan)
    pca_xy[valid_idx] = pca.fit_transform(X_scaled)

    # Auto-label clusters
    cluster_profiles = _auto_label(centroids, feature_names)

    return {
        "labels": labels_full.tolist(),
        "centroids": centroids.tolist(),
        "pca_xy": pca_xy.tolist(),
        "feature_names": feature_names,
        "cluster_profiles": cluster_profiles,
        "k_scores": k_scores,
        "n_clusters": n_clusters,
        "n_valid": len(valid_idx),
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
    }


def classify_new(new_features, model):
    """Classify a new trip using a trained clustering model.

    Returns cluster label (int).
    """
    _, feature_names, _ = prepare_features([new_features])
    X = np.zeros((1, model["kmeans"].n_features_in_))
    for i, f in enumerate(model["feature_names"]):
        X[0, i] = float(new_features.get(f, 0))
    X_scaled = model["scaler"].transform(X)
    return int(model["kmeans"].predict(X_scaled)[0])


def _auto_label(centroids, feature_names):
    """Assign human-readable labels to clusters based on centroid profiles.

    Returns dict: {cluster_id: {"label": str, "description": str, "profile": dict}}
    """
    # Build feature index for centroids
    profiles = {}

    for cid, centroid in enumerate(centroids):
        # Convert centroid to dict
        vals = {}
        for i, name in enumerate(feature_names):
            vals[name] = centroid[i]

        # Score against each style template
        best_style = None
        best_score = -999

        for style_name, template in STYLE_PROFILES.items():
            score = 0
            matched = 0
            for feat, direction in template.items():
                if feat in vals:
                    # direction: 1 = higher is better, -1 = lower is better, 0 = neutral
                    if direction > 0:
                        score += vals[feat]
                    elif direction < 0:
                        score -= vals[feat]
                    matched += 1
            if matched > 0:
                score /= matched
            if score > best_score:
                best_score = score
                best_style = style_name

        # Build a readable description
        desc_parts = []
        if vals.get("speed_mean", 0) > 45:
            desc_parts.append("均速较高")
        elif vals.get("speed_mean", 0) < 30:
            desc_parts.append("均速较低")

        if vals.get("pct_hard_accel", 0) > 10:
            desc_parts.append("急加速频繁")
        elif vals.get("pct_hard_accel", 0) < 5:
            desc_parts.append("加速温和")

        if vals.get("stop_pct", 0) > 25:
            desc_parts.append("停止时间多")
        elif vals.get("stop_pct", 0) < 10:
            desc_parts.append("几乎不停车")

        if vals.get("pct_high_speed", 0) > 40:
            desc_parts.append("高速为主")
        elif vals.get("pct_high_speed", 0) < 10:
            desc_parts.append("基本无高速")

        desc = "，".join(desc_parts) if desc_parts else "混合路况"

        profiles[cid] = {
            "label": best_style or f"模式{cid+1}",
            "description": desc,
            "centroid": {feature_names[i]: round(float(centroid[i]), 2)
                         for i in range(len(feature_names))},
        }

    return profiles
