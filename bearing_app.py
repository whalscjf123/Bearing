# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import os

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="공장 모터 고장 예측 시스템",
    page_icon="⚙️",
    layout="centered",
)

# ── 스타일 ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .title-box {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #00d4ff33;
        border-radius: 12px;
        padding: 2rem 2.5rem 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .title-box h1 {
        color: #00d4ff;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .title-box p { color: #8892b0; font-size: 0.95rem; }

    .result-box {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-normal  { background:#0d2b1a; border:1px solid #00ff8844; color:#00ff88; }
    .result-warning { background:#2b2000; border:1px solid #ffcc0044; color:#ffcc00; }
    .result-danger  { background:#2b0d0d; border:1px solid #ff444444; color:#ff4444; }

    .metric-card {
        background:#1a1f2e;
        border:1px solid #ffffff11;
        border-radius:8px;
        padding:1rem;
        text-align:center;
    }
    .metric-card .label { color:#8892b0; font-size:0.8rem; margin-bottom:0.3rem; }
    .metric-card .value { color:#e6edf3; font-size:1.3rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── GitHub raw URL 설정 ──────────────────────────────────────
GITHUB_USER = "whalscjf123"
GITHUB_REPO = "Bearing"
BRANCH      = "main"

def github_raw(filename):
    return f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{BRANCH}/{filename}"

# ── pkl 파일 로드 (캐시) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    errors = []

    # model.pkl
    try:
        r = requests.get(github_raw("model.pkl"), timeout=15)
        r.raise_for_status()
        model = joblib.load(io.BytesIO(r.content))
    except Exception as e:
        errors.append(f"model.pkl 로드 실패: {e}")
        model = None

    # scaler.pkl
    try:
        r = requests.get(github_raw("scaler.pkl"), timeout=15)
        r.raise_for_status()
        scaler = joblib.load(io.BytesIO(r.content))
    except Exception as e:
        errors.append(f"scaler.pkl 로드 실패: {e}")
        scaler = None

    return model, scaler, errors

# ── 군집 → 상태 해석 (k=3 기준) ─────────────────────────────
CLUSTER_INFO = {
    0: {"label": "⚠️ 주의 필요",   "desc": "미세한 이상 진동이 감지됩니다. 정기 점검을 권장합니다.", "cls": "result-warning"},
    1: {"label": "✅ 정상",         "desc": "베어링 상태가 양호합니다. 정상 운전 중입니다.",          "cls": "result-normal"},
    2: {"label": "🚨 고장 위험",    "desc": "심각한 진동 이상입니다. 즉시 점검이 필요합니다!",       "cls": "result-danger"},
}

# ── 헤더 ─────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
  <h1>⚙️ 공장 모터 고장 예측 시스템</h1>
  <p>베어링 진동 데이터를 입력하면 AI가 고장 위험도를 분석합니다</p>
</div>
""", unsafe_allow_html=True)

# ── 모델 로드 ────────────────────────────────────────────────
with st.spinner("🔄 AI 모델을 GitHub에서 불러오는 중..."):
    model, scaler, load_errors = load_artifacts()

if load_errors:
    for err in load_errors:
        st.error(err)
    st.stop()

st.success("✅ 모델 로드 완료")

st.divider()

# ── 입력 섹션 ────────────────────────────────────────────────
st.subheader("📊 베어링 진동 데이터 입력")
st.caption("실제 센서에서 측정한 값을 입력하세요")

col1, col2, col3 = st.columns(3)

with col1:
    max_val = st.number_input(
        "최댓값 (Max)",
        min_value=-10.0, max_value=10.0,
        value=0.0, step=0.01, format="%.4f",
        help="진동 신호의 최댓값"
    )
with col2:
    min_val = st.number_input(
        "최솟값 (Min)",
        min_value=-10.0, max_value=10.0,
        value=0.0, step=0.01, format="%.4f",
        help="진동 신호의 최솟값"
    )
with col3:
    rms_val = st.number_input(
        "실효값 RMS",
        min_value=0.0, max_value=5.0,
        value=0.1, step=0.001, format="%.4f",
        help="Root Mean Square — 진동 에너지 크기"
    )

st.divider()

# ── 예측 버튼 ────────────────────────────────────────────────
if st.button("🔍 고장 위험도 분석", use_container_width=True, type="primary"):

    # 입력 데이터프레임
    new_data = pd.DataFrame(
        [[max_val, min_val, rms_val]],
        columns=["최댓값", "최솟값", "실효값_RMS"]
    )

    # 표준화 → 예측
    scaled = scaler.transform(new_data)
    cluster = int(model.predict(scaled)[0])

    info = CLUSTER_INFO.get(cluster, {
        "label": f"군집 {cluster}",
        "desc": "해석 정보가 없습니다.",
        "cls": "result-warning"
    })

    # ── 결과 카드 ────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-box {info['cls']}">
        {info['label']}<br>
        <span style="font-size:0.9rem;font-weight:400;">{info['desc']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 입력값 요약")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="label">최댓값</div><div class="value">{max_val:.4f}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="label">최솟값</div><div class="value">{min_val:.4f}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="label">실효값 RMS</div><div class="value">{rms_val:.4f}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="label">예측 군집</div><div class="value">#{cluster}</div></div>', unsafe_allow_html=True)

# ── 푸터 ─────────────────────────────────────────────────────
st.divider()
st.caption("📌 CWRU Bearing Dataset 기반 K-Means(k=3) 군집 모델 | Streamlit Cloud 배포")
