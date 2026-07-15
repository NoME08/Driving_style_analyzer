# 🚗 Driving Style Analyzer

基于 OBD-II 数据的驾驶风格分析工具。通过 Car Scanner 采集的行车数据，自动分析驾驶模式、行程特征、评分和风格分类。

[English](./README.md)

## ✨ 功能

### 🌐 Web 界面（Streamlit）
- **首页** — 累计统计、最近行程概览、特征相关性矩阵
- **单程分析** — 上传 CSV，速度曲线（模式着色）、评分、行程明细、一键导出
- **行程对比** — 多选 2–4 行程，速度曲线叠加、雷达图、指标表、相关性热力图
- **驾驶画像** — 最近 vs 均值雷达图、特征/评分趋势、全部行程表、批量导出
- **风格分类** — K-Means 聚类 + PCA 可视化、自动命名、归类和聚类中心分析

### 🧠 分析引擎
| 模块 | 功能 |
|---|---|
| `features.py` | 26 维特征提取（速度、加速度、驾驶模式、发动机） |
| `style_classifier.py` | K-Means 无监督聚类 + PCA 降维 + `classify_new()` |
| `scorer.py` | 三维百分位评分（安全/平稳/经济）+ 综合分 |

### 📊 数据支持
- Car Scanner OBD 导出 CSV（半角分号分隔）
- 自动识别「车速」「发动机转速」「节气门位置」等 PID
- 兼容 GPS 和 OBD 两种速度源

## 🚀 快速开始

```bash
git clone https://github.com/yourusername/Driving_style_analyzer.git
cd Driving_style_analyzer

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`，拖拽 Car Scanner CSV 即可。

## 📁 项目结构

```
Driving_style_analyzer/
├── streamlit_app.py          # Web UI 入口（5 页面）
├── requirements.txt
├── src/
│   ├── data_loader.py        # CSV 解析（Car Scanner + UDDS）
│   ├── mode_detector.py      # 驾驶模式分类（stop/accel/decel/cruise）
│   ├── trip_analyzer.py      # 行程分割 + 统计
│   ├── features.py           # 26 维特征提取
│   ├── scorer.py             # 三维百分位评分引擎
│   ├── style_classifier.py   # K-Means 聚类 + PCA + 自动标签
│   ├── session_store.py      # 行程目录管理
│   ├── visualizer.py         # matplotlib 静态图（CLI 用）
│   └── main.py               # CLI 分析入口
├── data/                     # 行程 CSV 数据（gitignored）
└── output/                   # CLI 生成的图表（gitignored）
```

## 🔧 检测阈值

侧边栏可调：
- **急加速** — 默认 > +2.0 km/h/s
- **急减速** — 默认 < -2.0 km/h/s
- **停止判定** — 默认 < 1.5 km/h

## 📈 评分机制

每次行程与个人历史进行百分位比较：

| 维度 | 权重 | 依据 |
|---|---|---|
| 🛡️ 安全性 | 35% | 急减速频率、急加速频率、加速度波动 |
| 🏓 平稳性 | 35% | 加速度幅值、速度波动 |
| ⛽ 经济性 | 30% | 停止占比、巡航占比、RPM/节气门波动 |

> 50 = 历史中位数。评级：≥85 卓越，≥70 优秀，≥55 良好，≥40 一般，<40 待改善。

## 🧠 风格分类

K-Means 聚类将行程自动分为：
- **激进型** — 急加速频繁，城市激烈驾驶
- **高速巡航** — 高速占比高，相对平稳
- **拥堵通勤** — 停止占比极高，平均速度低

后续行程可通过 `classify_new()` 自动归类。

## 📦 依赖

- Python ≥ 3.9
- streamlit, plotly, pandas, numpy
- scikit-learn, matplotlib

## 📄 License

MIT
