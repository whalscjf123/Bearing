import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="베어링 군집 예측", page_icon="⚙️")

st.title("⚙️ 베어링 군집 예측 시스템")
st.markdown("베어링의 측정값을 입력하면 해당 베어링이 속하는 군집을 예측합니다.")

# ── 모델 & 스케일러 로드 ──────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    """저장된 모델과 스케일러를 로드합니다."""
    model, scaler = None, None

    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        st.warning("⚠️ model.pkl 파일을 찾을 수 없습니다. 같은 디렉터리에 배치해 주세요.")

    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        st.warning("⚠️ scaler.pkl 파일을 찾을 수 없습니다. 같은 디렉터리에 배치해 주세요.")

    return model, scaler


model, scaler = load_model_and_scaler()

# ── 사이드바: 파일 업로더(선택) ───────────────────────────────────────
with st.sidebar:
    st.header("📂 모델 파일 업로드 (선택)")
    uploaded_model  = st.file_uploader("model.pkl 업로드",  type=["pkl"])
    uploaded_scaler = st.file_uploader("scaler.pkl 업로드", type=["pkl"])

    if uploaded_model:
        model  = pickle.load(uploaded_model)
        st.success("모델 로드 완료 ✅")
    if uploaded_scaler:
        scaler = pickle.load(uploaded_scaler)
        st.success("스케일러 로드 완료 ✅")

# ── 입력 폼 ───────────────────────────────────────────────────────────
st.subheader("📊 베어링 측정값 입력")

col1, col2, col3 = st.columns(3)

with col1:
    max_val = st.number_input(
        "최댓값",
        value=0.0,
        format="%.6f",
        help="베어링 진동 신호의 최댓값"
    )

with col2:
    min_val = st.number_input(
        "최솟값",
        value=0.0,
        format="%.6f",
        help="베어링 진동 신호의 최솟값"
    )

with col3:
    rms_val = st.number_input(
        "실효값 (RMS)",
        value=0.0,
        format="%.6f",
        help="베어링 진동 신호의 RMS(Root Mean Square) 값"
    )

# ── 입력값 미리보기 ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 입력값 미리보기")

new_bearing = pd.DataFrame(
    [[max_val, min_val, rms_val]],
    columns=["최댓값", "최솟값", "실효값_RMS"]
)
st.dataframe(new_bearing, use_container_width=True)

# ── 예측 버튼 ─────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 군집 예측하기", type="primary", use_container_width=True)

if predict_btn:
    if model is None or scaler is None:
        st.error("❌ 모델 또는 스케일러가 로드되지 않았습니다. 사이드바에서 파일을 업로드해 주세요.")
    else:
        with st.spinner("예측 중..."):
            # 기존 학습 데이터 기준(scaler)으로 정규화 변환
            new_bearing_scaled = scaler.transform(new_bearing)

            # 모델을 통한 군집 예측
            pred_cluster = model.predict(new_bearing_scaled)

        st.success(f"✅ 이 베어링은 **{pred_cluster[0]}번 군집**에 속합니다.")

        # 정규화된 값도 함께 표시
        with st.expander("📐 정규화된 입력값 보기"):
            scaled_df = pd.DataFrame(
                new_bearing_scaled,
                columns=["최댓값 (정규화)", "최솟값 (정규화)", "실효값_RMS (정규화)"]
            )
            st.dataframe(scaled_df, use_container_width=True)

# ── 하단 안내 ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption("💡 model.pkl 과 scaler.pkl 파일을 앱과 같은 디렉터리에 두거나, 사이드바에서 직접 업로드하세요.")
