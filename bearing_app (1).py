import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io

st.set_page_config(page_title="베어링 군집 예측", page_icon="⚙️")

st.title("⚙️ 베어링 군집 예측 시스템")
st.markdown("베어링의 측정값을 입력하면 해당 베어링이 속하는 군집을 예측합니다.")

# ── GitHub Raw URL 설정 ───────────────────────────────────────────────
st.sidebar.header("🔗 GitHub 설정")

default_model_url = st.sidebar.text_input(
    "model.pkl GitHub Raw URL",
    value="https://raw.githubusercontent.com/your-username/your-repo/main/model.pkl",
    help="GitHub Raw URL 형식이어야 합니다. (raw.githubusercontent.com)",
)

default_scaler_url = st.sidebar.text_input(
    "scaler.pkl GitHub Raw URL",
    value="https://raw.githubusercontent.com/your-username/your-repo/main/scaler.pkl",
    help="GitHub Raw URL 형식이어야 합니다. (raw.githubusercontent.com)",
)

# ── GitHub에서 pkl 파일 로드 ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pkl_from_github(url: str):
    """GitHub Raw URL에서 pkl 파일을 다운로드하여 역직렬화합니다."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))


def try_load(label: str, url: str):
    try:
        with st.spinner(f"{label} 로딩 중..."):
            obj = load_pkl_from_github(url)
        st.sidebar.success(f"{label} 로드 완료 ✅")
        return obj
    except requests.exceptions.HTTPError as e:
        st.sidebar.error(f"❌ {label} HTTP 오류: {e}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error(f"❌ {label} 연결 실패. URL을 확인하세요.")
    except Exception as e:
        st.sidebar.error(f"❌ {label} 로드 실패: {e}")
    return None


# 사이드바 버튼으로 수동 로드 트리거
if st.sidebar.button("📥 모델 & 스케일러 불러오기", type="primary", use_container_width=True):
    st.session_state["model"]  = try_load("model.pkl",  default_model_url)
    st.session_state["scaler"] = try_load("scaler.pkl", default_scaler_url)

# session_state에서 모델/스케일러 참조
model  = st.session_state.get("model",  None)
scaler = st.session_state.get("scaler", None)

# 로드 상태 표시
col_status1, col_status2 = st.columns(2)
with col_status1:
    if model:
        st.info("🟢 모델 로드됨")
    else:
        st.warning("🔴 모델 미로드 — 사이드바에서 불러오기")
with col_status2:
    if scaler:
        st.info("🟢 스케일러 로드됨")
    else:
        st.warning("🔴 스케일러 미로드 — 사이드바에서 불러오기")

st.markdown("---")

# ── 입력 폼 ───────────────────────────────────────────────────────────
st.subheader("📊 베어링 측정값 입력")

col1, col2, col3 = st.columns(3)

with col1:
    max_val = st.number_input(
        "최댓값",
        value=0.0,
        format="%.6f",
        help="베어링 진동 신호의 최댓값",
    )
with col2:
    min_val = st.number_input(
        "최솟값",
        value=0.0,
        format="%.6f",
        help="베어링 진동 신호의 최솟값",
    )
with col3:
    rms_val = st.number_input(
        "실효값 (RMS)",
        value=0.0,
        format="%.6f",
        help="베어링 진동 신호의 RMS 값",
    )

# ── 입력값 미리보기 ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 입력값 미리보기")

new_bearing = pd.DataFrame(
    [[max_val, min_val, rms_val]],
    columns=["최댓값", "최솟값", "실효값_RMS"],
)
st.dataframe(new_bearing, use_container_width=True)

# ── 예측 버튼 ─────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 군집 예측하기", type="primary", use_container_width=True)

if predict_btn:
    if model is None or scaler is None:
        st.error("❌ 모델 또는 스케일러가 로드되지 않았습니다. 사이드바에서 URL 입력 후 '불러오기' 버튼을 눌러주세요.")
    else:
        with st.spinner("예측 중..."):
            new_bearing_scaled = scaler.transform(new_bearing)
            pred_cluster       = model.predict(new_bearing_scaled)

        st.success(f"✅ 이 베어링은 **{pred_cluster[0]}번 군집**에 속합니다.")

        with st.expander("📐 정규화된 입력값 보기"):
            scaled_df = pd.DataFrame(
                new_bearing_scaled,
                columns=["최댓값 (정규화)", "최솟값 (정규화)", "실효값_RMS (정규화)"],
            )
            st.dataframe(scaled_df, use_container_width=True)

# ── 하단 안내 ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption("💡 GitHub Raw URL 예시: https://raw.githubusercontent.com/username/repo/main/model.pkl")
