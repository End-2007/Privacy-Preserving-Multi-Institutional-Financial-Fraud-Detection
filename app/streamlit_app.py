"""Streamlit dashboard for the FedGuard Micro Project."""

from __future__ import annotations

import importlib
import html
import json
import math
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from src.common.config import REPORTS_DIR, ROOT
from src.common.inference import (
    DatasetArtifacts,
    load_dataset_artifacts,
    predict_transaction,
    transform_transaction,
)


DATASET_LABELS = {"paysim": "PaySim", "ieee_cis": "IEEE-CIS"}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
STRATEGY_LABELS = {
    "class_weight": "Class weights",
    "cap_smote": "Capped SMOTE",
    "cap_smote_enn": "Capped SMOTE-ENN",
}
SCORING_MODELS = {
    "XGBoost comparison model": "xgboost",
    "LightGBM selected model": None,
}
EXCLUDED_COLUMNS = {"isFraud", "isFlaggedFraud", "row_id", "TransactionID"}
USER_INPUT_FIELDS = {
    "paysim": {
        "numeric_nonnegative": (
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ),
        "required": ("type",),
    },
    "ieee_cis": {
        "numeric_nonnegative": ("TransactionAmt", "card1"),
        # Email domains and card network are naturally missing in IEEE-CIS.
        "required": ("ProductCD",),
    },
}


def _inject_visual_theme() -> None:
    """Add a CSS-only visual layer without changing the dashboard's content or behavior."""
    st.markdown(
        """
<style>
:root {
  --fg-bg: #07111f;
  --fg-panel: rgba(14, 27, 45, 0.78);
  --fg-line: rgba(148, 190, 229, 0.16);
  --fg-cyan: #62e6e0;
  --fg-blue: #6fa8ff;
  --fg-text: #e8f0fb;
  --fg-muted: #a8b9ce;
}

.stApp,
[data-testid="stAppViewContainer"] {
  color: var(--fg-text);
  background:
    radial-gradient(ellipse at 18% 5%, rgba(33, 94, 143, .22), transparent 37%),
    radial-gradient(ellipse at 88% 18%, rgba(20, 144, 151, .16), transparent 33%),
    linear-gradient(145deg, #07111f 0%, #0a1727 48%, #081321 100%);
}

[data-testid="stAppViewContainer"] { background-color: transparent; }
[data-testid="stMain"] { position: relative; z-index: 1; }
[data-testid="stSidebar"] {
  position: relative;
  z-index: 2;
  background: linear-gradient(180deg, rgba(10, 24, 41, .97), rgba(8, 18, 32, .94));
  border-right: 1px solid var(--fg-line);
  box-shadow: 12px 0 42px rgba(0, 0, 0, .17);
}
[data-testid="stSidebar"] > div:first-child { background: transparent; }
[data-testid="stMainBlockContainer"] {
  max-width: 1560px;
  padding-top: 2.2rem;
  padding-bottom: 4rem;
}

h1, h2, h3 {
  font-family: 'Segoe UI Variable Display', 'Segoe UI', sans-serif !important;
  letter-spacing: -0.035em;
}
h1 {
  background: linear-gradient(100deg, #f4f8ff 10%, #b9ddff 60%, #75eee1 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent !important;
  filter: drop-shadow(0 4px 24px rgba(68, 181, 224, .16));
}
p, label, [data-testid="stCaptionContainer"] { color: var(--fg-muted); }

/* Subtle glass panels that distinguish data and controls from the backdrop. */
[data-testid="stMetric"],
[data-testid="stForm"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
div[data-testid="stExpander"] {
  border: 1px solid var(--fg-line);
  border-radius: 14px;
  background: linear-gradient(145deg, rgba(20, 38, 59, .76), rgba(10, 23, 39, .73));
  box-shadow: 0 14px 38px rgba(0, 0, 0, .15), inset 0 1px rgba(255, 255, 255, .035);
}
[data-testid="stMetric"] { padding: 1rem 1.1rem; transition: transform .22s ease, border-color .22s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(98, 230, 224, .42); }
[data-testid="stMetricValue"] { color: #f3f8ff; }
[data-testid="stMetricLabel"] { color: #a9bed6; }
[data-testid="stDataFrame"], [data-testid="stTable"] { overflow: hidden; }

.fg-validation-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  margin: .7rem 0 1.5rem;
}
.fg-dataset-card {
  min-width: 0;
  padding: 1.15rem 1.2rem 1rem;
  border: 1px solid rgba(130, 185, 226, .2);
  border-radius: 16px;
  background: linear-gradient(145deg, rgba(15, 34, 55, .88), rgba(9, 22, 38, .82));
  box-shadow: 0 18px 44px rgba(0, 0, 0, .18), inset 0 1px rgba(255, 255, 255, .035);
  transform: perspective(900px) rotateX(.35deg);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.fg-dataset-card:hover { transform: perspective(900px) translateY(-3px) rotateX(0deg); border-color: rgba(98, 230, 224, .34); box-shadow: 0 22px 48px rgba(0, 0, 0, .25), inset 0 1px rgba(255, 255, 255, .06); }
.fg-dataset-card.paysim { border-top: 2px solid #58ded5; }
.fg-dataset-card.ieee { border-top: 2px solid #a78bfa; }
.fg-chart-heading { margin: 0 0 .95rem; color: #edf6ff; font-size: 1.05rem; font-weight: 700; }
.fg-chart-subtitle { margin: -.65rem 0 1.1rem; color: #9fb4cb; font-size: .82rem; }
.fg-chart-legend { display: flex; gap: 1rem; margin-bottom: .9rem; color: #b7c9dc; font-size: .78rem; }
.fg-legend-chip { display: inline-flex; align-items: center; gap: .4rem; }
.fg-legend-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.fg-legend-dot.ap { background: #58ded5; box-shadow: 0 0 12px rgba(88, 222, 213, .55); }
.fg-legend-dot.roc { background: #75aaff; box-shadow: 0 0 12px rgba(117, 170, 255, .55); }
.fg-chart-columns, .fg-chart-row { display: grid; grid-template-columns: minmax(84px, .9fr) minmax(70px, 1fr) 3.2rem minmax(70px, 1fr) 3.2rem; gap: .5rem; align-items: center; }
.fg-chart-columns { padding: 0 .15rem .45rem; border-bottom: 1px solid rgba(150, 185, 218, .14); color: #93a9c1; font-size: .69rem; text-transform: uppercase; letter-spacing: .06em; }
.fg-chart-row { padding: .72rem .15rem; border-bottom: 1px solid rgba(150, 185, 218, .09); }
.fg-chart-row:last-child { border-bottom: 0; }
.fg-model-name { overflow: hidden; color: #e3edf8; font-size: .8rem; font-weight: 600; text-overflow: ellipsis; white-space: nowrap; }
.fg-track { height: 9px; overflow: hidden; border-radius: 99px; background: rgba(164, 192, 220, .12); box-shadow: inset 0 1px 2px rgba(0, 0, 0, .28); }
.fg-fill { height: 100%; border-radius: inherit; }
.fg-fill.ap { background: linear-gradient(90deg, #188e9c, #58ded5); box-shadow: 0 0 14px rgba(88, 222, 213, .2); }
.fg-fill.roc { background: linear-gradient(90deg, #3a69bd, #75aaff); box-shadow: 0 0 14px rgba(117, 170, 255, .2); }
.fg-score { color: #dbe8f7; font-variant-numeric: tabular-nums; font-size: .76rem; text-align: right; }
@media (max-width: 850px) { .fg-validation-grid { grid-template-columns: 1fr; } }

.fg-shap-card {
  margin: .9rem 0 1.1rem;
  padding: 1rem 1.1rem;
  border: 1px solid rgba(128, 180, 222, .2);
  border-radius: 15px;
  background: linear-gradient(145deg, rgba(15, 34, 55, .88), rgba(9, 22, 38, .84));
  box-shadow: 0 16px 38px rgba(0, 0, 0, .18), inset 0 1px rgba(255, 255, 255, .035);
  transform: perspective(1000px) rotateX(.25deg);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.fg-shap-card:hover { transform: perspective(1000px) translateY(-2px) rotateX(0deg); border-color: rgba(128, 180, 222, .34); box-shadow: 0 21px 44px rgba(0, 0, 0, .24), inset 0 1px rgba(255, 255, 255, .06); }
.fg-shap-axis {
  display: grid;
  grid-template-columns: minmax(90px, 1fr) 34px minmax(90px, 1fr);
  align-items: center;
  margin: 0 4.5rem .65rem 0;
  color: #9db2c9;
  font-size: .72rem;
}
.fg-shap-axis span:nth-child(2) { color: #e5f1ff; text-align: center; }
.fg-shap-axis span:last-child { text-align: right; }
.fg-shap-row {
  display: grid;
  grid-template-columns: minmax(115px, 190px) minmax(120px, 1fr) 72px;
  align-items: center;
  gap: .75rem;
  min-height: 36px;
  padding: .2rem .35rem;
  border-radius: 8px;
  transition: background .18s ease, transform .18s ease;
}
.fg-shap-row:hover { background: rgba(131, 177, 220, .07); transform: translateX(2px); }
.fg-shap-feature {
  overflow: hidden;
  color: #dce8f5;
  font-size: .8rem;
  font-weight: 550;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fg-shap-track {
  position: relative;
  height: 19px;
  border-radius: 6px;
  background: linear-gradient(90deg, rgba(238, 117, 138, .045), rgba(163, 189, 216, .07) 49.8%, rgba(163, 189, 216, .07) 50.2%, rgba(70, 219, 194, .045));
}
.fg-shap-track::after {
  content: "";
  position: absolute;
  z-index: 2;
  left: 50%;
  top: 1px;
  bottom: 1px;
  width: 1px;
  background: rgba(208, 226, 244, .48);
}
.fg-shap-bar {
  position: absolute;
  top: 3px;
  bottom: 3px;
  border-radius: 4px;
  transition: filter .18s ease;
}
.fg-shap-row:hover .fg-shap-bar { filter: brightness(1.2); }
.fg-shap-bar.positive {
  left: 50%;
  background: linear-gradient(90deg, #1b9a9c, #5ce4cd);
  box-shadow: 0 0 15px rgba(69, 218, 199, .2);
}
.fg-shap-bar.negative {
  right: 50%;
  background: linear-gradient(270deg, #d47b85, #f0a0a4);
  box-shadow: 0 0 15px rgba(232, 128, 144, .17);
}
.fg-shap-value {
  font-size: .78rem;
  font-variant-numeric: tabular-nums;
  font-weight: 650;
  text-align: right;
}
.fg-shap-value.positive { color: #74e5d2; }
.fg-shap-value.negative { color: #f0a0a4; }
@media (max-width: 640px) {
  .fg-shap-card { padding: .8rem .65rem; }
  .fg-shap-axis { margin-right: 3.7rem; font-size: .65rem; }
  .fg-shap-row { grid-template-columns: minmax(90px, 130px) minmax(70px, 1fr) 58px; gap: .45rem; }
  .fg-shap-feature, .fg-shap-value { font-size: .7rem; }
}

/* Refined controls and focus states. */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  border: 1px solid rgba(98, 230, 224, .38);
  border-radius: 10px;
  color: #eaf7ff;
  background: linear-gradient(120deg, rgba(24, 93, 115, .75), rgba(32, 67, 113, .8));
  box-shadow: 0 7px 22px rgba(30, 130, 163, .16);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
  transform: translateY(-2px);
  border-color: rgba(98, 230, 224, .8);
  box-shadow: 0 10px 28px rgba(30, 160, 183, .26);
  color: white;
}
button:focus-visible, input:focus-visible, textarea:focus-visible {
  outline: 2px solid var(--fg-cyan) !important;
  outline-offset: 2px;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
  border-color: var(--fg-line);
  border-radius: 10px;
  background-color: rgba(16, 31, 49, .82);
}
[data-testid="stFileUploader"] section {
  border: 1px dashed rgba(98, 230, 224, .38);
  border-radius: 12px;
  background: rgba(13, 31, 49, .55);
}

/* Quietly color-code the existing notice blocks. */
[data-testid="stAlert"] {
  border: 1px solid var(--fg-line);
  border-radius: 12px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, .12);
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: 9px 9px 0 0;
  transition: color .18s ease, background .18s ease;
}
[data-testid="stTabs"] [role="tab"]:hover { background: rgba(78, 190, 205, .09); }
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--fg-cyan) !important;
}

/* Decorative, GPU-light 3D ambience: layered CSS spheres and orbit rings. */
.fedguard-atmosphere {
  position: fixed;
  inset: 0;
  z-index: 0;
  overflow: hidden;
  pointer-events: none;
  perspective: 1100px;
  transform-style: preserve-3d;
}
.fedguard-orb {
  position: absolute;
  width: 280px;
  height: 280px;
  border-radius: 50%;
  opacity: .18;
  filter: blur(1px);
  background: radial-gradient(circle at 32% 28%, rgba(185, 255, 251, .95), rgba(55, 198, 212, .52) 23%, rgba(35, 103, 185, .22) 55%, transparent 72%);
  box-shadow: inset -22px -28px 55px rgba(5, 13, 29, .55), 0 0 95px rgba(57, 184, 211, .14);
  animation: fg-float 20s ease-in-out infinite;
  transform-style: preserve-3d;
}
.fedguard-orb.one { top: 8%; right: 7%; }
.fedguard-orb.two {
  width: 190px; height: 190px; left: 16%; bottom: 4%; opacity: .12;
  animation-delay: -8s; animation-duration: 25s;
  background: radial-gradient(circle at 32% 26%, rgba(192, 219, 255, .9), rgba(80, 126, 238, .44) 28%, transparent 72%);
}
.fedguard-orbit {
  position: absolute;
  width: 440px;
  height: 150px;
  right: 1%; top: 13%;
  border: 1px solid rgba(98, 230, 224, .14);
  border-radius: 50%;
  transform: rotateX(68deg) rotateZ(-24deg);
  animation: fg-orbit 44s linear infinite;
}
.fedguard-orbit::before, .fedguard-orbit::after {
  content: "";
  position: absolute;
  inset: 18px 42px;
  border: 1px solid rgba(111, 168, 255, .12);
  border-radius: 50%;
}
.fedguard-orbit::after { inset: 38px 84px; border-color: rgba(98, 230, 224, .1); }
.fg-cube-wrap {
  position: absolute;
  width: 66px;
  height: 66px;
  opacity: .28;
  perspective: 700px;
  transform-style: preserve-3d;
}
.fg-cube-wrap.one { right: 11%; top: 43%; }
.fg-cube-wrap.two { left: 23%; bottom: 16%; width: 43px; height: 43px; opacity: .2; }
.fg-cube {
  position: relative;
  width: 100%;
  height: 100%;
  transform-style: preserve-3d;
  animation: fg-cube-turn 34s linear infinite;
}
.fg-cube-wrap.two .fg-cube { animation-duration: 42s; animation-direction: reverse; }
.fg-cube-face {
  position: absolute;
  inset: 0;
  border: 1px solid rgba(111, 223, 230, .68);
  background: linear-gradient(145deg, rgba(78, 210, 216, .1), rgba(91, 127, 228, .035));
  box-shadow: inset 0 0 15px rgba(73, 207, 222, .09), 0 0 12px rgba(73, 207, 222, .07);
  backface-visibility: visible;
}
.fg-cube-wrap.two .fg-cube-face { border-color: rgba(167, 139, 250, .72); background: linear-gradient(145deg, rgba(167, 139, 250, .12), rgba(91, 127, 228, .035)); }
.fg-cube-face.front { transform: translateZ(33px); }
.fg-cube-face.back { transform: rotateY(180deg) translateZ(33px); }
.fg-cube-face.right { transform: rotateY(90deg) translateZ(33px); }
.fg-cube-face.left { transform: rotateY(-90deg) translateZ(33px); }
.fg-cube-face.top { transform: rotateX(90deg) translateZ(33px); }
.fg-cube-face.bottom { transform: rotateX(-90deg) translateZ(33px); }
.fg-cube-wrap.two .fg-cube-face.front { transform: translateZ(21.5px); }
.fg-cube-wrap.two .fg-cube-face.back { transform: rotateY(180deg) translateZ(21.5px); }
.fg-cube-wrap.two .fg-cube-face.right { transform: rotateY(90deg) translateZ(21.5px); }
.fg-cube-wrap.two .fg-cube-face.left { transform: rotateY(-90deg) translateZ(21.5px); }
.fg-cube-wrap.two .fg-cube-face.top { transform: rotateX(90deg) translateZ(21.5px); }
.fg-cube-wrap.two .fg-cube-face.bottom { transform: rotateX(-90deg) translateZ(21.5px); }
.fg-particle {
  position: absolute;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: #78ede3;
  box-shadow: 0 0 13px 3px rgba(87, 224, 220, .28);
  animation: fg-particle-drift 8s ease-in-out infinite;
}
.fg-particle.p1 { top: 29%; right: 28%; }
.fg-particle.p2 { top: 61%; right: 7%; width: 3px; height: 3px; animation-delay: -3s; }
.fg-particle.p3 { bottom: 24%; left: 12%; background: #b39aff; box-shadow: 0 0 13px 3px rgba(167, 139, 250, .3); animation-delay: -5s; }
.fg-particle.p4 { top: 18%; left: 36%; width: 3px; height: 3px; animation-delay: -1s; }
.fedguard-grid {
  position: absolute;
  inset: 0;
  opacity: .12;
  background-image: linear-gradient(rgba(140, 187, 221, .1) 1px, transparent 1px), linear-gradient(90deg, rgba(140, 187, 221, .1) 1px, transparent 1px);
  background-size: 68px 68px;
  mask-image: linear-gradient(180deg, black, transparent 83%);
}
@keyframes fg-float {
  0%, 100% { transform: translate3d(0, 0, 0) rotateY(-14deg); }
  50% { transform: translate3d(-26px, 20px, 70px) rotateY(18deg); }
}
@keyframes fg-orbit {
  from { transform: rotateX(68deg) rotateZ(-24deg); }
  to { transform: rotateX(68deg) rotateZ(336deg); }
}
@keyframes fg-cube-turn {
  from { transform: rotateX(-24deg) rotateY(0deg) rotateZ(8deg); }
  to { transform: rotateX(336deg) rotateY(360deg) rotateZ(368deg); }
}
@keyframes fg-particle-drift {
  0%, 100% { transform: translate3d(0, 0, 0) scale(.75); opacity: .35; }
  50% { transform: translate3d(10px, -18px, 20px) scale(1.2); opacity: .9; }
}
@media (max-width: 700px) {
  .fg-cube-wrap.one { right: 5%; top: 48%; transform: scale(.72); }
  .fg-cube-wrap.two { left: 8%; bottom: 11%; transform: scale(.72); }
  .fg-particle.p4 { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
</style>
<div class="fedguard-atmosphere" aria-hidden="true">
  <div class="fedguard-grid"></div>
  <div class="fedguard-orb one"></div>
  <div class="fedguard-orb two"></div>
  <div class="fedguard-orbit"></div>
  <div class="fg-cube-wrap one"><div class="fg-cube"><i class="fg-cube-face front"></i><i class="fg-cube-face back"></i><i class="fg-cube-face right"></i><i class="fg-cube-face left"></i><i class="fg-cube-face top"></i><i class="fg-cube-face bottom"></i></div></div>
  <div class="fg-cube-wrap two"><div class="fg-cube"><i class="fg-cube-face front"></i><i class="fg-cube-face back"></i><i class="fg-cube-face right"></i><i class="fg-cube-face left"></i><i class="fg-cube-face top"></i><i class="fg-cube-face bottom"></i></div></div>
  <span class="fg-particle p1"></span><span class="fg-particle p2"></span><span class="fg-particle p3"></span><span class="fg-particle p4"></span>
</div>
""",
        unsafe_allow_html=True,
    )


def _configured_artifact_root() -> str:
    return os.environ.get("FEDGUARD_ARTIFACT_ROOT", str(ROOT))


def _resolve_artifact_root(artifact_root: str | Path, dataset: str) -> Path:
    """Resolve a shared bundle root or one of the separately extracted bundles."""
    root = Path(artifact_root).expanduser()
    bundle_name = f"stage1_transfer_{dataset}"
    candidates = [
        root,
        root / bundle_name,
        root / bundle_name / bundle_name,
        root / "stage1_transfer",
        root / "stage1_transfer" / bundle_name,
        root.parent / bundle_name,
        root.parent / bundle_name / bundle_name,
        root.parent / "stage1_transfer",
        root.parent / "stage1_transfer" / bundle_name,
    ]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "models" / dataset).is_dir()
            and (candidate / "data" / "processed" / dataset).is_dir()
        ):
            return candidate
    return root


@st.cache_resource(show_spinner=False)
def cached_artifacts(
    dataset: str,
    artifact_root: str = str(ROOT),
    model_family: str | None = None,
) -> DatasetArtifacts:
    return load_dataset_artifacts(
        dataset,
        root=_resolve_artifact_root(artifact_root, dataset),
        model_family=model_family,
    )


@st.cache_data(show_spinner=False)
def load_sample_transactions(
    dataset: str,
    limit: int = 50,
    artifact_root: str = str(ROOT),
) -> pd.DataFrame:
    """Load a small batch of validation rows; never read the held-out test split."""
    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if limit < 1:
        raise ValueError("limit must be positive")

    resolved_root = _resolve_artifact_root(artifact_root, dataset)
    path = resolved_root / "data" / "processed" / dataset / "validation.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Validation samples were not found: {path}")

    parquet_file = pq.ParquetFile(path)
    batch = next(parquet_file.iter_batches(batch_size=limit), None)
    if batch is None:
        raise ValueError(f"Validation samples are empty: {path}")
    frame = batch.to_pandas()
    frame = frame.drop(columns=list(EXCLUDED_COLUMNS.intersection(frame.columns)))
    return frame.reset_index(drop=True)


def load_metrics(dataset: str) -> dict:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    path = REPORTS_DIR / f"{dataset}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Metrics report was not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def metrics_table() -> pd.DataFrame:
    """Final held-out test metrics for the selected LightGBM models."""
    rows = []
    for dataset in DATASET_LABELS:
        report = load_metrics(dataset)
        metrics = report.get("test_metrics", {})
        rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Model": MODEL_LABELS.get(
                    report.get("model"), report.get("model", "unknown")
                ),
                "Validation-selected threshold": report.get(
                    "threshold", metrics.get("threshold")
                ),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1"),
                "Average Precision (AP)": metrics.get("pr_auc"),
                "ROC-AUC": metrics.get("roc_auc"),
                "FPR": metrics.get("fpr"),
            }
        )
    return pd.DataFrame(rows)


def validation_comparison_table(artifact_root: str | Path = ROOT) -> pd.DataFrame:
    """Return validation comparisons, using saved thresholded family metrics if present."""
    rows = []
    for dataset, dataset_label in DATASET_LABELS.items():
        comparison = load_metrics(dataset).get("comparison", [])
        best_by_model = {}
        for candidate in comparison:
            model = candidate.get("model")
            validation = candidate.get("validation", {})
            if model not in MODEL_LABELS:
                continue
            try:
                average_precision = float(validation["pr_auc"])
                roc_auc = float(validation["roc_auc"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(average_precision) or not math.isfinite(roc_auc):
                continue
            current = best_by_model.get(model)
            if current is None or average_precision > current["average_precision"]:
                best_by_model[model] = {
                    "average_precision": average_precision,
                    "roc_auc": roc_auc,
                    "strategy": candidate.get("strategy", "unknown"),
                }

        missing_models = [model for model in MODEL_LABELS if model not in best_by_model]
        if missing_models:
            raise ValueError(
                f"Validation comparison is incomplete for {dataset}: {', '.join(missing_models)}"
            )

        for model, result in best_by_model.items():
            validation = {}
            threshold = None
            strategy = result["strategy"]
            metadata_path = (
                _resolve_artifact_root(artifact_root, dataset)
                / "models"
                / dataset
                / "comparison"
                / model
                / "metadata.json"
            )
            if metadata_path.exists():
                try:
                    with metadata_path.open("r", encoding="utf-8") as stream:
                        family_metadata = json.load(stream)
                    if family_metadata.get("dataset") == dataset:
                        validation = family_metadata.get("validation_metrics", {})
                        threshold = family_metadata.get("threshold")
                        strategy = family_metadata.get("strategy", strategy)
                except (OSError, json.JSONDecodeError):
                    # Keep the committed ranking metrics as a safe fallback.
                    validation = {}

            rows.append(
                {
                    "Dataset": dataset_label,
                    "Model": MODEL_LABELS[model],
                    "Best validation strategy": STRATEGY_LABELS.get(
                        strategy, str(strategy)
                    ),
                    "Validation-selected threshold": threshold,
                    "Validation Average Precision (AP)": result["average_precision"],
                    "Validation ROC-AUC": result["roc_auc"],
                    "Precision": validation.get("precision"),
                    "Recall": validation.get("recall"),
                    "F1": validation.get("f1"),
                    "False-positive rate": validation.get("fpr"),
                    "Split": "Validation only",
                }
            )
    return pd.DataFrame(rows)


def _render_validation_chart(table: pd.DataFrame, dataset_label: str) -> None:
    """Render the selected dataset's validation AP and ROC-AUC chart."""
    rows = table.loc[table["Dataset"] == dataset_label].sort_values(
        "Validation Average Precision (AP)", ascending=False
    )
    card_class = "paysim" if dataset_label == "PaySim" else "ieee"
    chart_rows = []
    for _, row in rows.iterrows():
        ap = min(1.0, max(0.0, float(row["Validation Average Precision (AP)"])))
        roc = min(1.0, max(0.0, float(row["Validation ROC-AUC"])))
        model = str(row["Model"])
        chart_rows.append(
            "<div class='fg-chart-row'>"
            f"<div class='fg-model-name'>{model}</div>"
            f"<div class='fg-track'><div class='fg-fill ap' style='width:{ap * 100:.2f}%'></div></div>"
            f"<div class='fg-score'>{ap:.3f}</div>"
            f"<div class='fg-track'><div class='fg-fill roc' style='width:{roc * 100:.2f}%'></div></div>"
            f"<div class='fg-score'>{roc:.3f}</div>"
            "</div>"
        )
    card = (
        f"<section class='fg-dataset-card {card_class}'>"
        f"<h4 class='fg-chart-heading'>{dataset_label}</h4>"
        "<div class='fg-chart-columns'><span>Model</span><span>AP</span><span></span><span>ROC-AUC</span><span></span></div>"
        + "".join(chart_rows)
        + "</section>"
    )
    st.markdown(
        "<div class='fg-chart-legend'>"
        "<span class='fg-legend-chip'><i class='fg-legend-dot ap'></i>Average Precision (AP)</span>"
        "<span class='fg-legend-chip'><i class='fg-legend-dot roc'></i>ROC-AUC</span>"
        "</div>"
        + card,
        unsafe_allow_html=True,
    )


def _risk_label(score: float, threshold: float) -> str:
    """Return a heuristic interface category relative to the chosen threshold."""
    if score >= threshold:
        return "High"
    if score >= threshold * 0.5:
        return "Medium"
    return "Low"


def _signed_contribution_bars(contributions, *, bar_width: int = 20) -> str:
    """Format local SHAP values as dependency-light signed text bars."""
    rows = []
    for item in contributions:
        try:
            feature = str(item.source_feature)
            value = float(item.shap_value)
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        rows.append((feature, value))

    if not rows:
        return ""

    largest = max(abs(value) for _, value in rows) or 1.0
    lines = ["Negative contributions  <-       0       ->  Positive contributions"]
    for feature, value in rows:
        width = max(1, int(round(abs(value) / largest * bar_width)))
        bar = "█" * min(width, bar_width)
        left = bar.rjust(bar_width) if value < 0 else " " * bar_width
        right = " " * bar_width if value < 0 else bar.ljust(bar_width)
        lines.append(f"{feature[:32]:32} {left}|{right} {value:+.4f}")
    return "\n".join(lines)


def _signed_contribution_chart(contributions) -> str:
    """Build a dependency-light, accessible signed SHAP bar chart."""
    rows = []
    for item in contributions:
        try:
            feature = str(item.source_feature)
            value = float(item.shap_value)
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            rows.append((feature, value))
    if not rows:
        return ""

    largest = max(abs(value) for _, value in rows) or 1.0
    rendered_rows = []
    for feature, value in rows:
        direction = "positive" if value >= 0 else "negative"
        width = min(50.0, abs(value) / largest * 50.0)
        safe_feature = html.escape(feature, quote=True)
        rendered_rows.append(
            f"<div class='fg-shap-row' role='listitem' "
            f"aria-label='{safe_feature}, SHAP contribution {value:+.4f}'>"
            f"<div class='fg-shap-feature' title='{safe_feature}'>{safe_feature}</div>"
            "<div class='fg-shap-track' aria-hidden='true'>"
            f"<div class='fg-shap-bar {direction}' style='width:{width:.2f}%'></div>"
            "</div>"
            f"<div class='fg-shap-value {direction}'>{value:+.4f}</div>"
            "</div>"
        )
    return (
        "<div class='fg-shap-card'>"
        "<div class='fg-shap-axis'><span>Negative contributions</span><span>0</span>"
        "<span>Positive contributions</span></div>"
        "<div role='list'>"
        + "".join(rendered_rows)
        + "</div></div>"
    )


def _feature_columns(artifacts: DatasetArtifacts) -> list[str]:
    names = getattr(artifacts.preprocessor, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]
    metadata = artifacts.feature_metadata
    groups = (
        metadata.get("numeric_features", []),
        metadata.get("frequency_encoded_features", []),
        metadata.get("one_hot_features", []),
    )
    columns = list(dict.fromkeys(str(column) for group in groups for column in group))
    if not columns:
        raise ValueError(f"No input feature columns found for {artifacts.dataset}")
    return columns


def _prepare_transaction(
    frame: pd.DataFrame,
    artifacts: DatasetArtifacts,
) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Upload exactly one transaction; received {len(frame)} rows.")
    frame = frame.drop(columns=list(EXCLUDED_COLUMNS.intersection(frame.columns)))
    columns = _feature_columns(artifacts)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:12])
        raise ValueError(
            f"CSV is missing {len(missing)} required model input column(s): {preview}"
        )
    return frame[columns].iloc[0].copy()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _text_input_default(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def validate_transaction_input(dataset: str, transaction: pd.Series) -> list[str]:
    """Return user-facing validation messages for the visible key fields."""
    if dataset not in USER_INPUT_FIELDS:
        return [f"Unsupported dataset: {dataset}"]

    errors: list[str] = []
    rules = USER_INPUT_FIELDS[dataset]
    for field in rules["required"]:
        if not _text_input_default(transaction.get(field)).strip():
            errors.append(f"{field} is required.")

    for field in rules["numeric_nonnegative"]:
        try:
            number = float(transaction.get(field))
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
            continue
        if not math.isfinite(number):
            errors.append(f"{field} must be finite.")
        elif number < 0:
            errors.append(f"{field} cannot be negative.")
    return errors


def _derive_paysim_features(transaction: pd.Series) -> pd.Series:
    """Recalculate PaySim features affected by edited balances."""
    amount = _safe_float(transaction.get("amount"))
    old_orig = _safe_float(transaction.get("oldbalanceOrg"))
    new_orig = _safe_float(transaction.get("newbalanceOrig"))
    old_dest = _safe_float(transaction.get("oldbalanceDest"))
    new_dest = _safe_float(transaction.get("newbalanceDest"))
    transaction["orig_balance_delta"] = new_orig - old_orig
    transaction["dest_balance_delta"] = new_dest - old_dest
    transaction["error_balance_orig"] = new_orig + amount - old_orig
    transaction["error_balance_dest"] = old_dest + amount - new_dest
    transaction["orig_zero_after"] = int(new_orig == 0)
    transaction["log_amount"] = math.log1p(max(amount, 0.0))
    return transaction


def _load_explanation(dataset: str, transaction: pd.Series, artifacts: DatasetArtifacts):
    try:
        module = importlib.import_module("src.common.explain")
        explain_transaction = getattr(module, "explain_transaction")
        return explain_transaction(
            dataset,
            transaction,
            artifacts=artifacts,
            max_features=10,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("The SHAP explanation module is unavailable.") from exc


def _load_explanation_or_report(
    dataset: str,
    transaction: pd.Series,
    artifacts: DatasetArtifacts,
):
    """Keep prediction available when the optional SHAP explanation fails."""
    try:
        return _load_explanation(dataset, transaction, artifacts)
    except Exception as exc:
        st.warning(f"SHAP explanation could not be generated: {exc}")
        return None


def _show_explanation(explanation) -> None:
    if explanation is None:
        st.info("The score is available, but a SHAP explanation could not be generated.")
        return

    st.subheader("Explanation of this prediction")
    st.caption(
        "SHAP contributions are on the model's raw output scale. They describe this "
        "prediction; they do not establish causation."
    )
    st.caption(f"SHAP base output: {explanation.base_value:.6f}")
    for reason in explanation.plain_language_reasons:
        st.write(f"- {reason}")

    contributions = explanation.contributions
    if not contributions:
        st.info("No finite SHAP contributions were available to display.")
        return

    chart = _signed_contribution_chart(contributions)
    if chart:
        st.markdown(chart, unsafe_allow_html=True)
    rows = [
        {
            "Model feature": item.feature_name,
            "Source feature": item.source_feature,
            "Input value": _text_input_default(item.value),
            "SHAP contribution": item.shap_value,
            "Direction": {
                "increases_risk": "Pushes toward fraud class",
                "decreases_risk": "Pushes away from fraud class",
                "no_effect": "No directional effect",
            }.get(item.direction, item.direction),
        }
        for item in contributions
    ]
    st.dataframe(
        pd.DataFrame(rows).style.format({"SHAP contribution": "{:+.4f}"}),
        width="stretch",
        hide_index=True,
    )


def _score_transaction(
    dataset: str,
    transaction: pd.Series,
    artifacts: DatasetArtifacts,
    model_label: str,
    artifact_root: str,
) -> None:
    errors = validate_transaction_input(dataset, transaction)
    if errors:
        st.error("Please correct the following input(s):")
        for error in errors:
            st.write(f"- {error}")
        return

    try:
        # This also checks the complete input schema before prediction.
        transform_transaction(transaction, artifacts)
        result = predict_transaction(dataset, transaction, artifacts=artifacts)
    except Exception as exc:
        if "access violation" in str(exc).lower():
            st.error(
                "The selected model's native prediction library failed on this machine. "
                "Choose the XGBoost comparison model for transaction scoring; the "
                "selected LightGBM test results remain available in Model results."
            )
            return
        st.error(f"Prediction failed: {exc}")
        return

    explanation = _load_explanation_or_report(dataset, transaction, artifacts)

    st.session_state[f"fedguard_result_{dataset}"] = {
        "dataset": dataset,
        "model_label": model_label,
        "artifact_root": str(_resolve_artifact_root(artifact_root, dataset).resolve()),
        "transaction": transaction.copy(),
        "result": result,
        "explanation": explanation,
    }


def _render_demo_editor(dataset: str, samples: pd.DataFrame):
    index = st.selectbox(
        "Validation reference transaction",
        options=list(samples.index),
        format_func=lambda value: f"Example {value + 1} (validation split; label excluded)",
        key=f"{dataset}_reference_index",
    )
    if st.button("Load demonstration row", key=f"{dataset}_load_demo"):
        st.session_state[f"{dataset}_demo_row"] = samples.loc[index].copy()

    transaction = st.session_state.get(f"{dataset}_demo_row")
    if transaction is None:
        st.info("Choose a validation example and load it, or upload a one-row CSV instead.")
        return None, False
    transaction = transaction.copy()

    with st.form(f"{dataset}_demo_form"):
        if dataset == "paysim":
            types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
            current_type = str(transaction.get("type", "PAYMENT"))
            type_index = types.index(current_type) if current_type in types else 3
            col1, col2, col3 = st.columns(3)
            with col1:
                transaction["type"] = st.selectbox(
                    "Transaction type", types, index=type_index
                )
                transaction["amount"] = st.number_input(
                    "Amount",
                    min_value=0.0,
                    value=_safe_float(transaction.get("amount")),
                    step=100.0,
                )
            with col2:
                transaction["oldbalanceOrg"] = st.number_input(
                    "Sender balance before",
                    min_value=0.0,
                    value=_safe_float(transaction.get("oldbalanceOrg")),
                    step=100.0,
                )
                transaction["newbalanceOrig"] = st.number_input(
                    "Sender balance after",
                    min_value=0.0,
                    value=_safe_float(transaction.get("newbalanceOrig")),
                    step=100.0,
                )
            with col3:
                transaction["oldbalanceDest"] = st.number_input(
                    "Receiver balance before",
                    min_value=0.0,
                    value=_safe_float(transaction.get("oldbalanceDest")),
                    step=100.0,
                )
                transaction["newbalanceDest"] = st.number_input(
                    "Receiver balance after",
                    min_value=0.0,
                    value=_safe_float(transaction.get("newbalanceDest")),
                    step=100.0,
                )
            transaction = _derive_paysim_features(transaction)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                transaction["TransactionAmt"] = st.number_input(
                    "Transaction amount",
                    min_value=0.0,
                    value=_safe_float(transaction.get("TransactionAmt")),
                    step=10.0,
                )
                transaction["ProductCD"] = st.text_input(
                    "Product code",
                    value=_text_input_default(transaction.get("ProductCD", "")),
                )
            with col2:
                transaction["card1"] = st.number_input(
                    "Card identifier",
                    min_value=0.0,
                    value=_safe_float(transaction.get("card1")),
                    step=1.0,
                )
                card4 = st.text_input(
                    "Card network",
                    value=_text_input_default(transaction.get("card4", "")),
                )
                transaction["card4"] = card4.strip() or pd.NA
            with col3:
                purchaser = st.text_input(
                    "Purchaser email domain",
                    value=_text_input_default(transaction.get("P_emaildomain", "")),
                )
                recipient = st.text_input(
                    "Recipient email domain",
                    value=_text_input_default(transaction.get("R_emaildomain", "")),
                )
                transaction["P_emaildomain"] = purchaser.strip() or pd.NA
                transaction["R_emaildomain"] = recipient.strip() or pd.NA

        submitted = st.form_submit_button("Score transaction")
    return transaction, submitted


def _render_csv_input(dataset: str, artifacts: DatasetArtifacts, artifact_root: str):
    try:
        samples = load_sample_transactions(dataset, limit=1, artifact_root=artifact_root)
        columns = _feature_columns(artifacts)
        template = samples.loc[[0], columns]
        template_csv = template.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV input template",
            data=template_csv,
            file_name=f"{dataset}_transaction_template.csv",
            mime="text/csv",
            help="The template starts with a validation example and excludes its label.",
        )
    except Exception as exc:
        st.warning(f"CSV template is unavailable: {exc}")

    uploaded = st.file_uploader(
        "Upload a one-row transaction CSV",
        type=["csv"],
        key=f"{dataset}_transaction_csv",
    )
    if uploaded is None:
        st.info("Upload one transaction, or choose the validation-example input mode.")
        return None, False

    try:
        transaction = _prepare_transaction(pd.read_csv(uploaded), artifacts)
    except Exception as exc:
        st.error(f"CSV input could not be used: {exc}")
        return None, False
    st.caption("Target and identifier columns are ignored; the saved model input schema is used.")
    return transaction, st.button("Score transaction", key=f"{dataset}_score_csv")


def _display_scored_result(
    dataset: str,
    transaction: pd.Series,
    model_label: str,
    artifact_root: str,
) -> None:
    state = st.session_state.get(f"fedguard_result_{dataset}")
    if not state:
        return
    if state["model_label"] != model_label:
        return
    if state["artifact_root"] != str(_resolve_artifact_root(artifact_root, dataset).resolve()):
        return
    if not state["transaction"].equals(transaction):
        return

    result = state["result"]
    st.subheader("Prediction result")
    score_col, threshold_col, decision_col, category_col = st.columns(4)
    with score_col:
        st.metric("Fraud model score (uncalibrated)", f"{result.fraud_probability:.4f}")
    with threshold_col:
        st.metric("Validation-selected threshold", f"{result.threshold:.4f}")
    with decision_col:
        st.metric(
            "Decision at threshold",
            "Flagged for review" if result.predicted_label else "Below threshold",
        )
    with category_col:
        st.metric(
            "Interface category (heuristic)",
            _risk_label(result.fraud_probability, result.threshold),
        )
    st.caption(
        f"Scoring model: {result.model_name}. The displayed score is not a calibrated "
        "real-world fraud probability. Low, medium, and high are heuristic interface categories."
    )

    transaction_preview = transaction.head(24).rename("input value").to_frame()
    with st.expander("Transaction input preview", expanded=False):
        st.dataframe(transaction_preview, width="stretch")
        if len(transaction) > 24:
            st.caption(f"Showing 24 of {len(transaction)} model input fields.")
    _show_explanation(state["explanation"])


def _render_transaction_tab(dataset: str, artifact_root: str, model_label: str, model_family):
    label = DATASET_LABELS[dataset]
    try:
        artifacts = cached_artifacts(dataset, artifact_root, model_family)
    except Exception as exc:
        st.warning(
            f"{label} model bundle is not available at the selected artifact folder. "
            "Set the local Stage 1 artifact folder in the sidebar and ensure it contains "
            f"models/{dataset} and data/processed/{dataset}. Details: {exc}"
        )
        return

    st.caption(
        f"Scoring model: {model_label}. Use a one-row CSV or a validation example. "
        "Validation labels are excluded from model input."
    )
    input_mode = st.radio(
        "Transaction input",
        ("Upload one-row CSV", "Use validation example"),
        horizontal=True,
        key=f"{dataset}_input_mode",
    )

    transaction = None
    submitted = False
    if input_mode == "Upload one-row CSV":
        transaction, submitted = _render_csv_input(dataset, artifacts, artifact_root)
    else:
        try:
            samples = load_sample_transactions(
                dataset, limit=50, artifact_root=artifact_root
            )
        except Exception as exc:
            st.warning(
                f"Validation examples are unavailable for {label}: {exc}. "
                "The model-results and scope tabs remain available."
            )
            return
        transaction, submitted = _render_demo_editor(dataset, samples)

    if transaction is None:
        return

    if submitted:
        _score_transaction(
            dataset, transaction, artifacts, model_label, artifact_root
        )
    _display_scored_result(dataset, transaction, model_label, artifact_root)


def _render_model_results(dataset: str, artifact_root: str) -> None:
    dataset_label = DATASET_LABELS[dataset]
    st.subheader(f"{dataset_label} validation ranking")
    st.caption(
        "Average Precision and ROC-AUC are validation ranking metrics shown on the same 0–1 scale."
    )
    all_validation = None
    try:
        all_validation = validation_comparison_table(artifact_root)
        _render_validation_chart(all_validation, dataset_label)
    except Exception as exc:
        st.warning(f"Validation comparison chart could not be loaded: {exc}")

    st.subheader(f"{dataset_label} validation model comparison")
    st.caption(
        "One best validation-ranking result per model family. These are validation results, "
        "not held-out test results. Average Precision is abbreviated AP."
    )
    try:
        table = all_validation
        if table is None:
            table = validation_comparison_table(artifact_root)
        table = table.loc[table["Dataset"] == dataset_label].sort_values(
            "Validation Average Precision (AP)", ascending=False
        )
        st.dataframe(
            table.style.format(
                {
                    "Validation-selected threshold": "{:.4f}",
                    "Validation Average Precision (AP)": "{:.4f}",
                    "Validation ROC-AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "False-positive rate": "{:.3%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        if table["Validation-selected threshold"].isna().any():
            st.info(
                "The committed reports contain validation Average Precision and ROC-AUC. "
                "Extract the local model bundle to show each model's selected threshold, "
                "precision, recall, F1, and false-positive rate."
            )
    except Exception as exc:
        st.warning(f"Validation comparison could not be loaded: {exc}")

    st.subheader("Selected LightGBM model: held-out test results")
    st.caption(
        "The final test split was evaluated once after model and threshold selection. "
        "Test results were not used to tune either."
    )
    try:
        table = metrics_table()
        table = table.loc[table["Dataset"] == dataset_label]
        st.dataframe(
            table.style.format(
                {
                    "Validation-selected threshold": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "Average Precision (AP)": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "FPR": "{:.3%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    except Exception as exc:
        st.warning(f"Held-out test metrics could not be loaded: {exc}")


def _render_scope(dataset: str) -> None:
    label = DATASET_LABELS[dataset]
    st.subheader("Scope and limitations")
    st.markdown(
        f"""
- **Separate datasets:** {label} uses its own model, preprocessing pipeline, feature names, and validation-selected threshold. Results are not pooled with the other dataset.
- **Validation and test:** interactive examples come from the validation split. The held-out test metrics are displayed separately and were evaluated once.
- **Heuristic interface categories:** Low, Medium, and High are presentation categories relative to the selected threshold. They are not calibrated risk bands.
- **Model scores:** displayed scores are not calibrated real-world fraud probabilities and are not proof of fraud.
- **PaySim:** this is synthetic data. Its post-transaction balance fields make the demonstration retrospective, not pre-transaction prevention.
- **IEEE-CIS:** coded or anonymized fields retain their supplied names; the dashboard does not assign unsupported business meanings to them.
- **Centralized demonstration:** this local app does not implement federated learning, differential privacy, secure aggregation, production controls, or transaction blocking.
"""
    )


def main() -> None:
    st.set_page_config(page_title="FedGuard", layout="wide")
    _inject_visual_theme()

    with st.sidebar:
        st.subheader("FedGuard controls")
        artifact_root = st.text_input(
            "Local Stage 1 artifact folder",
            value=_configured_artifact_root(),
            help=(
                "Choose the extracted local bundle containing models/ and "
                "data/processed/. These files stay out of Git."
            ),
        )
        st.caption(
            "The local model bundle is intentionally excluded from Git. "
            "Enter the common folder or the extracted folder for either dataset; "
            "separately extracted PaySim and IEEE-CIS bundles are detected automatically."
        )
        selected_dataset_label = st.selectbox(
            "Dataset / simulated institution",
            options=list(DATASET_LABELS.values()),
        )
        dataset = next(
            name for name, label in DATASET_LABELS.items() if label == selected_dataset_label
        )
        model_label = st.selectbox(
            "Transaction scoring model",
            options=list(SCORING_MODELS),
            index=0,
            help=(
                "XGBoost uses its validation-comparison model and threshold. LightGBM "
                "uses the selected Stage 1 model. Final test metrics remain the selected "
                "LightGBM results."
            ),
        )
        model_family = SCORING_MODELS[model_label]

    st.title("FedGuard | Micro Project")
    st.subheader("Transaction scoring and model review")
    st.warning(
        "Research demonstration only. It does not block transactions or make customer decisions."
    )
    if dataset == "paysim":
        st.info(
            "PaySim is a synthetic mobile-money dataset. Its balance fields are observed "
            "after transactions, so the demonstration is retrospective."
        )
    else:
        st.info(
            "IEEE-CIS is a separate card/e-commerce dataset with coded and anonymized "
            "fields. Its model and metrics are reported separately from PaySim."
        )

    score_tab, results_tab, scope_tab = st.tabs(
        ["Transaction score + SHAP", "Model results", "Scope and limitations"]
    )
    with score_tab:
        st.subheader("Score one transaction")
        _render_transaction_tab(dataset, artifact_root.strip(), model_label, model_family)
    with results_tab:
        _render_model_results(dataset, artifact_root.strip())
    with scope_tab:
        _render_scope(dataset)


if __name__ == "__main__":
    main()
