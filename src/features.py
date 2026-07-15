"""Feature extraction — converts a driving trip DataFrame into a feature vector.

Reusable across the Streamlit UI (profile page, comparison) and future ML pipeline.
"""

import numpy as np


def extract_features(df, accel_th=2.0, brake_th=-2.0, stop_th=1.5):
    """Extract driving-style features from a single trip DataFrame.

    Expects df with columns: time_sec, speed_kmh, acceleration, driving_mode
    Optionally: rpm, throttle_pct, lat, lon

    Returns a flat dict of scalar features.
    """
    n = len(df)
    if n == 0:
        return {}

    dur_min = df["time_sec"].max() / 60

    # ── Speed ──────────────────────────────────────────
    spd = df["speed_kmh"]
    speed_mean = float(spd.mean())
    speed_max = float(spd.max())
    speed_std = float(spd.std())

    pct_low = float((spd <= 20).sum() / n * 100)
    pct_mid = float(((spd > 20) & (spd <= 60)).sum() / n * 100)
    pct_high = float((spd > 60).sum() / n * 100)

    # ── Acceleration ───────────────────────────────────
    acc = df["acceleration"]
    acc_std = float(acc.std())
    acc_mag_mean = float(acc.abs().mean())

    pos_acc = acc[acc > 0]
    neg_acc = acc[acc < 0]
    acc_pos_mean = float(pos_acc.mean()) if len(pos_acc) > 0 else 0.0
    acc_neg_mean = float(neg_acc.mean()) if len(neg_acc) > 0 else 0.0

    hard_accel_pct = float((acc > accel_th).sum() / n * 100)
    hard_brake_pct = float((acc < brake_th).sum() / n * 100)
    hard_accel_count = int((acc > accel_th).sum())
    hard_brake_count = int((acc < brake_th).sum())

    # Aggressiveness index (0-100)
    aggro = float((hard_accel_count + hard_brake_count) / n * 100)

    # ── Driving modes ──────────────────────────────────
    modes = df["driving_mode"].value_counts()
    stop_pct = float(modes.get("stop", 0) / n * 100)
    accel_pct = float(modes.get("accel", 0) / n * 100)
    decel_pct = float(modes.get("decel", 0) / n * 100)
    cruise_pct = float(modes.get("cruise", 0) / n * 100)

    # ── Engine / OBD ───────────────────────────────────
    rpm_mean = _safe_stat(df, "rpm", "mean")
    rpm_std = _safe_stat(df, "rpm", "std")
    thr_mean = _safe_stat(df, "throttle_pct", "mean")
    thr_std = _safe_stat(df, "throttle_pct", "std")

    # ── Trip fragmentation ────────────────────────────
    trip_ids = df[df.get("trip_id", -1) > 0]["trip_id"].nunique() if "trip_id" in df.columns else None

    return {
        # Meta
        "duration_min": round(dur_min, 1),
        "data_points": n,
        # Speed
        "speed_mean": round(speed_mean, 1),
        "speed_max": round(speed_max, 0),
        "speed_std": round(speed_std, 1),
        "pct_low_speed": round(pct_low, 1),
        "pct_mid_speed": round(pct_mid, 1),
        "pct_high_speed": round(pct_high, 1),
        # Acceleration
        "acc_std": round(acc_std, 2),
        "acc_mag_mean": round(acc_mag_mean, 2),
        "acc_pos_mean": round(acc_pos_mean, 2),
        "acc_neg_mean": round(acc_neg_mean, 2),
        "pct_hard_accel": round(hard_accel_pct, 1),
        "pct_hard_brake": round(hard_brake_pct, 1),
        "hard_accel_count": hard_accel_count,
        "hard_brake_count": hard_brake_count,
        "aggressiveness": round(aggro, 1),
        # Modes
        "stop_pct": round(stop_pct, 1),
        "accel_pct": round(accel_pct, 1),
        "decel_pct": round(decel_pct, 1),
        "cruise_pct": round(cruise_pct, 1),
        # Engine
        "rpm_mean": round(rpm_mean, 0) if rpm_mean is not None else None,
        "rpm_std": round(rpm_std, 0) if rpm_std is not None else None,
        "throttle_mean": round(thr_mean, 1) if thr_mean is not None else None,
        "throttle_std": round(thr_std, 1) if thr_std is not None else None,
        # Trips
        "trip_count": int(trip_ids) if trip_ids is not None else None,
    }


# ── Helpers ────────────────────────────────────────────


def _safe_stat(df, col, method):
    """Return df[col].method() or None if column missing/empty."""
    if col in df.columns and df[col].notna().any():
        return float(getattr(df[col], method)())
    return None


# ── Named feature groups (for UI radar / comparison) ──

RADAR_DIMS = [
    ("speed_mean",       "均速"),
    ("speed_std",        "速度波动"),
    ("pct_high_speed",   "高速占比"),
    ("acc_std",          "加速波动"),
    ("acc_mag_mean",     "加速幅值"),
    ("pct_hard_accel",   "急加速%"),
    ("pct_hard_brake",   "急减速%"),
    ("stop_pct",         "停止占比"),
    ("cruise_pct",       "巡航占比"),
    ("aggressiveness",   "激烈度"),
]
