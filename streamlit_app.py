#!/usr/bin/env python3
"""Driving Style Analyzer — Streamlit Web App (v3)."""

import sys, os, io, csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.data_loader import load_data
from src.mode_detector import detect_driving_modes
from src.trip_analyzer import identify_trips, calculate_trip_statistics
from src.features import extract_features, RADAR_DIMS
from src.session_store import scan_data_files, load_catalog, update_entry, delete_file
from src.style_classifier import cluster_trips, CORE_FEATURES, FEATURE_NAMES
from src.scorer import score_trip, score_all, score_label

# ── Page config ─────────────────────────────────────────
st.set_page_config(page_title="Driving Style Analyzer", layout="wide")

# ── Color palette ───────────────────────────────────────
C = {
    "blue": "#007aff", "red": "#ff3b30", "orange": "#ff9500",
    "green": "#34c759", "gray": "#aeaeb2", "purple": "#af52de",
    "bg": "rgba(0,0,0,0)",
    "stop": "#aeaeb2", "accel": "#ff3b30",
    "decel": "#ff9500", "cruise": "#007aff",
}
MODE_COLORS = {"stop": C["stop"], "accel": C["accel"], "decel": C["decel"], "cruise": C["cruise"]}
MODE_NAMES = {"stop": "停止", "accel": "加速", "decel": "减速", "cruise": "巡航"}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

def _to_csv_bytes(features_list, scores_list, filenames):
    out = io.StringIO()
    keys = list(features_list[0].keys()) if features_list else []
    score_keys = ["composite", "safety", "smoothness", "efficiency"]
    writer = csv.DictWriter(out, fieldnames=["filename"] + keys + score_keys)
    writer.writeheader()
    for i, feat in enumerate(features_list):
        row = {"filename": filenames[i] if i < len(filenames) else ""}
        row.update({k: feat.get(k, "") for k in keys})
        if i < len(scores_list):
            row.update(scores_list[i])
        writer.writerow(row)
    return out.getvalue().encode("utf-8")


def _ensure_catalog():
    """Analyze all uncatalogued files, return (files, entries)."""
    files = scan_data_files()
    catalog = load_catalog()
    for f in files:
        if f["name"] not in catalog:
            try:
                res = run_pipeline(f["path"], accel_th, brake_th, stop_th)
                update_entry(f["name"], res["summary"], res["features"])
            except Exception:
                pass
    catalog = load_catalog()
    entries = [(fname, catalog[fname])
               for fname in sorted(catalog.keys(), reverse=True)
               if fname in {f["name"] for f in files}]
    return files, entries


# ══════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_pipeline(file_path, accel_th, brake_th, stop_th):
    df = load_data(file_path)
    df = detect_driving_modes(df, accel_threshold=accel_th,
                              brake_threshold=brake_th, stop_threshold=stop_th)
    df = identify_trips(df)
    trips_df, summary = calculate_trip_statistics(df)
    feat = extract_features(df, accel_th=accel_th, brake_th=brake_th, stop_th=stop_th)

    step = max(1, len(df) // 500)
    df_sample = df.iloc[::step]
    profile = [{"t": float(r["time_sec"]), "speed": float(r["speed_kmh"]),
                "mode": r["driving_mode"]} for _, r in df_sample.iterrows()]

    modes = df["driving_mode"].value_counts()
    n = len(df)
    mode_dist = {}
    for m in ["stop", "accel", "decel", "cruise"]:
        cnt = int(modes.get(m, 0))
        mode_dist[m] = {"count": cnt, "pct": round(cnt / n * 100, 1)}

    trip_list = [{
        "id": int(t["trip_id"]),
        "duration_sec": round(float(t["duration_sec"]), 0),
        "distance_km": round(float(t["distance_km"]), 3),
        "avg_speed": round(float(t["avg_speed_kmh"]), 1),
        "max_speed": round(float(t["max_speed_kmh"]), 1),
    } for _, t in trips_df.iterrows()]

    return {
        "profile": profile, "mode_dist": mode_dist, "trips": trip_list,
        "summary": {
            "duration_min": round(df["time_sec"].max() / 60, 1),
            "distance_km": round(summary["total_distance_km"], 3),
            "avg_speed": round(df["speed_kmh"].mean(), 1),
            "max_speed": round(df["speed_kmh"].max(), 1),
            "data_points": len(df),
            "trip_count": int(summary["total_trips"]),
        },
        "features": feat,
    }


# ══════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════

def _plotly_defaults(fig, height):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor=C["bg"], paper_bgcolor=C["bg"])
    return fig


def make_speed_chart(profile, height=380):
    df_p = pd.DataFrame(profile)
    fig = go.Figure()
    for mode, color in MODE_COLORS.items():
        mask = df_p[df_p["mode"] == mode]
        if len(mask) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=mask["t"] / 60, y=mask["speed"], mode="markers",
            marker=dict(color=color, size=3, opacity=0.6),
            name=MODE_NAMES[mode],
            hovertemplate="%{y:.1f} km/h<extra>" + MODE_NAMES[mode] + "</extra>",
        ))
    return _plotly_defaults(fig, height)


def make_mode_pie(mode_dist, height=280):
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["停止", "加速", "减速", "巡航"],
        values=[mode_dist[m]["pct"] for m in ["stop", "accel", "decel", "cruise"]],
        marker_colors=[C["stop"], C["accel"], C["decel"], C["cruise"]],
        hole=0.55, textinfo="label+percent", sort=False,
    ))
    return _plotly_defaults(fig, height)


def make_compare_speed_chart(profiles, labels, height=380):
    dash_styles = [None, "dash", "dot", "dashdot"]
    colors = [C["blue"], C["orange"], C["green"], C["purple"]]
    fig = go.Figure()
    for i, (prof, label) in enumerate(zip(profiles, labels)):
        df_p = pd.DataFrame(prof)
        t_range = df_p["t"].max() - df_p["t"].min() + 0.001
        t_pct = (df_p["t"] - df_p["t"].min()) / t_range * 100
        fig.add_trace(go.Scatter(
            x=t_pct, y=df_p["speed"], mode="lines",
            line=dict(color=colors[i % len(colors)], width=2,
                      dash=dash_styles[i % len(dash_styles)]),
            name=label, hovertemplate="%{y:.1f} km/h<extra>" + label + "</extra>",
        ))
    return _plotly_defaults(fig, height)


def make_radar_chart(features_list, labels, height=450):
    dim_keys = [d[0] for d in RADAR_DIMS]
    dim_names = [d[1] for d in RADAR_DIMS]
    colors_pal = px.colors.qualitative.Set1
    fig = go.Figure()
    for i, feat in enumerate(features_list):
        vals = [feat.get(k, 0) or 0 for k in dim_keys]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=dim_names + [dim_names[0]], fill="toself",
            name=labels[i] if i < len(labels) else f"#{i+1}",
            opacity=0.35,
            line=dict(color=colors_pal[i % len(colors_pal)], width=2),
        ))
    fig.update_layout(height=height, margin=dict(l=40, r=40, t=40, b=40),
                      polar=dict(radialaxis=dict(visible=True)),
                      plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                      legend=dict(orientation="h", y=-0.1))
    return fig


def make_trend_chart(entries, metric_key, metric_name, height=320):
    vals, labels = [], []
    for fname, entry in entries:
        feat = entry.get("features", {})
        if metric_key in feat and feat[metric_key] is not None:
            labels.append(fname.replace(".csv", "").replace("2026-", ""))
            vals.append(feat[metric_key])
    if len(vals) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(vals))), y=vals, mode="lines+markers",
        line=dict(color=C["blue"], width=2), marker=dict(size=8),
        hovertemplate="%{y}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
        yaxis_title=metric_name, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
    )
    return fig


def make_corr_heatmap(features_list, labels, height=400):
    dim_keys = [d[0] for d in RADAR_DIMS]
    data = {}
    for i, feat in enumerate(features_list):
        label = labels[i] if i < len(labels) else f"#{i+1}"
        data[label] = [feat.get(k, 0) or 0 for k in dim_keys]
    df_corr = pd.DataFrame(data).T.corr()
    dim_names = [d[1] for d in RADAR_DIMS]
    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values, x=dim_names, y=dim_names,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(df_corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=50),
                      plot_bgcolor=C["bg"], paper_bgcolor=C["bg"])
    return fig


def make_score_trend_chart(vals, dim_name, height=260):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(vals))), y=vals, mode="lines+markers",
        line=dict(color=C["blue"], width=2), marker=dict(size=6),
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color=C["gray"], opacity=0.5)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(range=[0, 100], title=dim_name),
                      plot_bgcolor=C["bg"], paper_bgcolor=C["bg"])
    return fig


# ══════════════════════════════════════════════════════════
# UI: Sidebar
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🚗 Driving Style")
    st.caption("Analyzer v3")
    st.divider()

    page = st.radio(
        "导航",
        ["🏠 首页", "📤 单程分析", "🆚 行程对比", "📊 驾驶画像", "🧠 风格分类"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("⚙️ 检测阈值")
    accel_th = st.slider("急加速 (km/h/s)", 0.5, 6.0, 2.0, 0.25)
    brake_th = st.slider("急减速 (km/h/s)", -6.0, -0.5, -2.0, 0.25)
    stop_th = st.slider("停止判定 (km/h)", 0.5, 5.0, 1.5, 0.25)


# ══════════════════════════════════════════════════════════
# PAGE 0: 首页
# ══════════════════════════════════════════════════════════

if page == "🏠 首页":
    st.header("🏠 驾驶风格分析仪")

    files, entries = _ensure_catalog()

    if not entries:
        st.info("还没有行程数据。去「📤 单程分析」上传你的第一个 Car Scanner OBD CSV 吧。")
        st.markdown("""
        ### 🚀 快速开始
        1. 用 **Car Scanner** App 连接 OBD-II 记录驾驶行程
        2. 导出 CSV（需包含「车速」PID）
        3. 拖拽到「📤 单程分析」页面
        4. 自动生成分析报告、评分、风格分类
        """)
    else:
        all_feats = [e[1].get("features", {}) for e in entries]
        all_feats = [f for f in all_feats if f]
        sc_list = score_all(all_feats)

        total_trips = len(all_feats)
        total_dist = sum(
            (f.get("duration_min", 0) or 0) * (f.get("speed_mean", 0) or 0) / 60
            for f in all_feats)
        total_dur = sum(f.get("duration_min", 0) or 0 for f in all_feats)
        avg_score = float(np.mean([s.get("composite", 50) for s in sc_list])) if sc_list else 50

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📚 累计行程", f"{total_trips} 次")
        c2.metric("🛣️ 累计里程", f"{total_dist:.0f} km")
        c3.metric("⏱️ 累计时长", f"{total_dur:.0f} min")
        lbl, _ = score_label(avg_score)
        c4.metric("🏆 平均评分", f"{avg_score:.0f}",
                  delta=lbl, delta_color="normal" if avg_score >= 55 else "inverse")

        st.divider()

        # Recent trips
        st.subheader("📋 最近行程")
        recent = entries[:5]
        for fname, entry in recent:
            feat = entry.get("features", {})
            sm = entry.get("summary", {})
            cols = st.columns([3, 1, 1, 1, 1])
            with cols[0]:
                st.write(f"**{fname.replace('.csv','').replace('2026-','')}**")
            with cols[1]:
                st.caption(f"⏱️ {feat.get('duration_min','?')}min")
            with cols[2]:
                st.caption(f"📏 {sm.get('distance_km','?')}km")
            with cols[3]:
                st.caption(f"⚡ {feat.get('aggressiveness','?')}%")
            with cols[4]:
                idx_list = [i for i, (fn, _) in enumerate(entries) if fn == fname]
                if idx_list and idx_list[0] < len(sc_list):
                    sc = sc_list[idx_list[0]]
                    sl, _ = score_label(sc.get("composite", 50))
                    st.caption(f"🏆 {sc.get('composite',50):.0f} {sl}")

        st.divider()

        # Feature correlation
        if len(all_feats) >= 3:
            st.subheader("📈 特征相关性")
            all_labels = [e[0].replace(".csv", "").replace("2026-", "")[:20] for e in entries]
            st.plotly_chart(make_corr_heatmap(all_feats, all_labels, height=420),
                            use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 1: 单程分析
# ══════════════════════════════════════════════════════════

elif page == "📤 单程分析":
    st.header("📤 单程分析")

    uploaded = st.file_uploader(
        "拖拽 Car Scanner OBD CSV 文件", type=["csv"],
        key="uploader", label_visibility="collapsed",
    )

    if uploaded:
        tmp_path = os.path.join(DATA_DIR, uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("分析中…"):
            try:
                result = run_pipeline(tmp_path, accel_th, brake_th, stop_th)
            except Exception as e:
                st.error(f"分析失败: {e}")
                st.stop()
        st.session_state["last_result"] = result
        st.session_state["last_filename"] = uploaded.name
        update_entry(uploaded.name, result["summary"], result["features"])
        run_pipeline.clear()

    result = st.session_state.get("last_result")
    filename = st.session_state.get("last_filename")

    if result:
        s, f = result["summary"], result["features"]

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⏱️ 时长", f"{s['duration_min']} min")
        c2.metric("📏 里程", f"{s['distance_km']} km")
        c3.metric("🚀 均速", f"{s['avg_speed']} km/h")
        c4.metric("🔺 最高速", f"{s['max_speed']} km/h")
        aggro = f.get('aggressiveness', 0)
        c5.metric("⚡ 激烈度", f"{aggro}%",
                  delta=None if aggro < 10 else "⚠️ 偏高")

        # Scoring
        _, entries = _ensure_catalog()
        hist_feats = [e[1].get("features", {}) for e in entries]
        hist_feats = [hf for hf in hist_feats if hf] or [f]
        sc = score_trip(f, hist_feats)
        cl, _ = score_label(sc["composite"])

        st.divider()
        st.caption("📊 驾驶评分（相对个人历史百分位）")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("🏆 综合评分", f"{sc['composite']}",
                   delta=cl, delta_color="normal" if sc["composite"] >= 55 else "inverse")
        sc2.metric("🛡️ 安全性", f"{sc['safety']}")
        sc3.metric("🏓 平稳性", f"{sc['smoothness']}")
        sc4.metric("⛽ 经济性", f"{sc['efficiency']}")

        # Speed chart
        st.subheader("速度曲线（驾驶模式着色）")
        st.plotly_chart(make_speed_chart(result["profile"]), use_container_width=True)

        # Mode pie + trips
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("驾驶模式分布")
            st.plotly_chart(make_mode_pie(result["mode_dist"]), use_container_width=True)
        with c2:
            st.subheader("行程明细")
            if result["trips"]:
                df_t = pd.DataFrame(result["trips"])
                df_t["时长"] = df_t["duration_sec"].apply(lambda x: f"{int(x//60)}分{int(x%60)}秒")
                df_t = df_t.rename(columns={"id": "#", "distance_km": "里程(km)",
                                            "avg_speed": "均速", "max_speed": "最高速"})
                st.dataframe(df_t[["#", "时长", "里程(km)", "均速", "最高速"]],
                             hide_index=True, use_container_width=True)
            else:
                st.caption("无行程段")

        # Export
        csv_data = _to_csv_bytes([f], [sc], [filename])
        st.download_button("📥 导出 CSV", data=csv_data,
                           file_name=filename.replace(".csv", "_analysis.csv"),
                           mime="text/csv")
    else:
        st.info("👆 上传 Car Scanner OBD CSV 文件开始分析。")


# ══════════════════════════════════════════════════════════
# PAGE 2: 行程对比
# ══════════════════════════════════════════════════════════

elif page == "🆚 行程对比":
    st.header("🆚 行程对比")

    files = scan_data_files()
    if len(files) < 2:
        st.info("需要至少 2 个 CSV 文件才能对比。")
    else:
        file_names = [f["name"] for f in files]
        selected = st.multiselect(
            "选择 2-4 个行程进行对比",
            file_names,
            default=file_names[:min(2, len(file_names))],
            max_selections=4,
        )

        if len(selected) < 2:
            st.info("请至少选择 2 个行程")
        else:
            results = []
            for name in selected:
                path = os.path.join(DATA_DIR, name)
                try:
                    res = run_pipeline(path, accel_th, brake_th, stop_th)
                    results.append((name, res))
                except Exception as e:
                    st.error(f"{name}: {e}")

            if len(results) < 2:
                st.stop()

            labels = [n.replace(".csv", "").replace("2026-", "") for n, _ in results]
            feats = [r["features"] for _, r in results]
            profs = [r["profile"] for _, r in results]

            # Speed overlay
            st.subheader("速度曲线叠加（归一化时间轴）")
            st.plotly_chart(make_compare_speed_chart(profs, labels), use_container_width=True)

            # Radar
            st.subheader("多维度雷达图")
            st.plotly_chart(make_radar_chart(feats, labels), use_container_width=True)

            # Metrics table
            st.subheader("指标对比")
            comp_keys = [
                ("duration_min", "时长(min)", "{:.1f}"),
                ("speed_mean", "均速", "{:.1f}"),
                ("speed_max", "最高速", "{:.0f}"),
                ("aggressiveness", "激烈度%", "{:.1f}"),
                ("stop_pct", "停止%", "{:.0f}"),
                ("cruise_pct", "巡航%", "{:.0f}"),
                ("hard_accel_count", "急加速", "{}"),
                ("hard_brake_count", "急减速", "{}"),
                ("acc_std", "加速波动σ", "{:.2f}"),
            ]
            rows = []
            for key, display, fmt in comp_keys:
                row = {"指标": display}
                for i, ft in enumerate(feats):
                    val = ft.get(key, "—")
                    row[labels[i]] = fmt.format(val) if val != "—" else "—"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Correlation heatmap
            if len(selected) >= 3:
                st.divider()
                st.subheader("特征相关性矩阵")
                st.plotly_chart(make_corr_heatmap(feats, labels, height=420),
                                use_container_width=True)

            # Export
            sc_list = score_all(feats)
            csv_data = _to_csv_bytes(feats, sc_list, selected)
            st.download_button("📥 导出对比 CSV", data=csv_data,
                               file_name="trip_comparison.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════
# PAGE 3: 驾驶画像
# ══════════════════════════════════════════════════════════

elif page == "📊 驾驶画像":
    st.header("📊 驾驶画像")

    files, entries = _ensure_catalog()

    if not entries:
        st.info("还没有行程数据。去「单程分析」上传 CSV 吧。")
    else:
        all_feats = [e[1].get("features", {}) for e in entries]
        all_feats = [f for f in all_feats if f]
        all_labels = [e[0].replace(".csv", "").replace("2026-", "") for e in entries]

        # Radar: latest vs average
        st.subheader("最近行程 vs 个人画像")
        if len(all_feats) >= 2:
            avg_feat = {}
            for k in all_feats[0]:
                vv = [f.get(k) for f in all_feats if f.get(k) is not None]
                avg_feat[k] = float(np.mean(vv)) if vv else None
            st.plotly_chart(
                make_radar_chart([all_feats[0], avg_feat],
                                 [f"最近: {all_labels[0]}", "个人均值"]),
                use_container_width=True)
        elif len(all_feats) == 1:
            st.plotly_chart(make_radar_chart([all_feats[0]], [all_labels[0]]),
                            use_container_width=True)

        # Feature trends
        st.subheader("特征趋势")
        trend_metrics = [
            ("aggressiveness", "激烈度 %"), ("speed_mean", "均速 km/h"),
            ("acc_std", "加速度波动 σ"), ("stop_pct", "停止占比 %"),
            ("cruise_pct", "巡航占比 %"),
        ]
        for row_start in [0, 3]:
            row_metrics = trend_metrics[row_start:row_start + 3]
            cols = st.columns(len(row_metrics))
            for i, (key, name) in enumerate(row_metrics):
                with cols[i]:
                    chart = make_trend_chart(entries, key, name)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.caption(f"{name}: 数据不足")

        # Score trends
        st.divider()
        st.subheader("评分趋势")
        sc_list = score_all(all_feats)
        score_dims = [
            ("composite", "🏆 综合"), ("safety", "🛡️ 安全"),
            ("smoothness", "🏓 平稳"), ("efficiency", "⛽ 经济"),
        ]
        cols = st.columns(4)
        for i, (dim, name) in enumerate(score_dims):
            with cols[i]:
                vals = [s.get(dim, 50) for s in sc_list]
                if len(vals) >= 2:
                    st.plotly_chart(make_score_trend_chart(vals, name),
                                    use_container_width=True)
                else:
                    st.caption(f"{name}: 数据不足")

        # Feature table
        st.divider()
        st.subheader("全部行程特征")
        df_feat = pd.DataFrame(all_feats)
        df_feat.insert(0, "行程", [l[:30] for l in all_labels])
        display_cols = ["行程", "duration_min", "speed_mean", "speed_max",
                        "aggressiveness", "stop_pct", "cruise_pct",
                        "hard_accel_count", "hard_brake_count"]
        display_cols = [c for c in display_cols if c in df_feat.columns]
        st.dataframe(df_feat[display_cols], hide_index=True, use_container_width=True)

        # Export
        csv_data = _to_csv_bytes(all_feats, sc_list, [e[0] for e in entries])
        st.download_button("📥 导出全部 CSV", data=csv_data,
                           file_name="all_trips_analysis.csv", mime="text/csv")

        # File management
        st.divider()
        with st.expander("📁 管理行程文件"):
            catalog = load_catalog()
            for f in files:
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    entry = catalog.get(f["name"], {})
                    dur = entry.get("features", {}).get("duration_min", "?")
                    dist = entry.get("summary", {}).get("distance_km", "?")
                    st.write(f"**{f['name']}** · {dur}min · {dist}km · {f['size_kb']}KB")
                with c2:
                    if st.button("📋 查看", key=f"open_{f['name']}"):
                        try:
                            res = run_pipeline(f["path"], accel_th, brake_th, stop_th)
                            st.session_state["last_result"] = res
                            st.session_state["last_filename"] = f["name"]
                            st.success("已加载，切换到「单程分析」查看")
                        except Exception:
                            st.error("加载失败")
                with c3:
                    if st.button("🗑️", key=f"del_{f['name']}"):
                        delete_file(f["name"])
                        st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE 4: 风格分类
# ══════════════════════════════════════════════════════════

elif page == "🧠 风格分类":
    st.header("🧠 驾驶风格分类")
    st.caption("基于 K-Means 无监督聚类，自动发现驾驶风格模式。")

    files, entries = _ensure_catalog()

    if len(entries) < 3:
        st.info("至少需要 3 个行程才能聚类。")
    else:
        features_list = [e[1].get("features", {}) for e in entries]
        features_list = [f for f in features_list if f]
        filenames = [e[0].replace(".csv", "").replace("2026-", "") for e in entries]

        if len(features_list) < 3:
            st.info("至少需要 3 个有效特征。")
        else:
            c1, _ = st.columns([1, 3])
            with c1:
                k_choice = st.selectbox("聚类数 K", [2, 3, 4, 5], index=1)

            with st.spinner("聚类中…"):
                cluster_result = cluster_trips(features_list, n_clusters=k_choice)

            if cluster_result is None:
                st.error("聚类失败。")
            else:
                # PCA scatter
                st.subheader("PCA 行程分布 (2D 降维)")
                cluster_colors = ["#007aff", "#ff9500", "#34c759", "#af52de", "#ff3b30"]
                pca_xy = np.array(cluster_result["pca_xy"])
                labels_arr = np.array(cluster_result["labels"])

                fig_pca = go.Figure()
                for cid in sorted(set(lbl for lbl in labels_arr if lbl >= 0)):
                    mask = labels_arr == cid
                    prof = cluster_result["cluster_profiles"].get(cid, {})
                    cname = prof.get("label", f"簇{cid}")
                    fig_pca.add_trace(go.Scatter(
                        x=pca_xy[mask, 0], y=pca_xy[mask, 1],
                        mode="markers+text",
                        name=f"{cname} ({mask.sum()}个)",
                        text=[filenames[i] for i, m in enumerate(mask) if m],
                        textposition="top center", textfont=dict(size=10),
                        marker=dict(size=14, color=cluster_colors[cid % len(cluster_colors)],
                                    line=dict(width=1, color="white")),
                        hovertemplate="%{text}<extra></extra>",
                    ))
                fig_pca.update_layout(
                    height=420, margin=dict(l=20, r=20, t=10, b=20),
                    xaxis_title="PC1", yaxis_title="PC2",
                    plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                    legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_pca, use_container_width=True)

                # K info
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("聚类数", cluster_result["n_clusters"])
                    st.caption(f"有效: {cluster_result['n_valid']} 行程")
                    if cluster_result["k_scores"]:
                        ss = "  ".join(f"K={k}:{v:.3f}" for k, v in
                                       sorted(cluster_result["k_scores"].items()))
                        st.caption(f"轮廓系数: {ss}")

                # Profiles
                st.divider()
                st.subheader("各风格画像")
                cols = st.columns(len(cluster_result["cluster_profiles"]))
                for i, (cid, prof) in enumerate(cluster_result["cluster_profiles"].items()):
                    with cols[i]:
                        color = cluster_colors[int(cid) % len(cluster_colors)]
                        st.markdown(f"<h4 style='color:{color}'>{prof['label']}</h4>",
                                    unsafe_allow_html=True)
                        st.caption(prof["description"])
                        trips_here = [filenames[j] for j, lbl in enumerate(labels_arr)
                                      if lbl == int(cid)]
                        st.caption(f"**{len(trips_here)} 个:**")
                        for t in trips_here:
                            st.caption(f"· {t}")

                # Centroids
                st.divider()
                st.subheader("聚类中心对比")
                rows = []
                for cid, prof in cluster_result["cluster_profiles"].items():
                    row = {"风格": prof["label"]}
                    for fk in cluster_result["feature_names"]:
                        row[FEATURE_NAMES.get(fk, fk)] = prof["centroid"].get(fk, "—")
                    rows.append(row)
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                # Assignments
                st.divider()
                st.subheader("行程归类明细")
                trip_rows = []
                for i, (fname, feat) in enumerate(zip(filenames, features_list)):
                    lbl = labels_arr[i]
                    style = "—"
                    if lbl >= 0:
                        style = cluster_result["cluster_profiles"].get(lbl, {}).get("label", f"簇{lbl}")
                    trip_rows.append({
                        "行程": fname, "风格": style,
                        "时长(min)": feat.get("duration_min", "—"),
                        "均速": feat.get("speed_mean", "—"),
                        "激烈度%": feat.get("aggressiveness", "—"),
                    })
                st.dataframe(pd.DataFrame(trip_rows), hide_index=True, use_container_width=True)
