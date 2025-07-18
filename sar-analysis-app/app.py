import streamlit as st
import pandas as pd
from utils import load_data, find_activity_cliffs, generate_hypothesis, draw_molecule
import plotly.express as px

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 기반 SAR 분석 시스템",
    page_icon="🧪",
    layout="wide"
)

# --- 사이드바 ---
with st.sidebar:
    st.title("SAR 분석 설정")
    st.markdown("---")

    # 데이터 업로드
    uploaded_file = st.file_uploader("SAR 데이터(.csv)를 업로드하세요.", type="csv")
    use_sample_data = st.checkbox("샘플 데이터 사용", value=True)

    st.markdown("---")
    
    # 분석 파라미터
    st.subheader("분석 파라미터")
    similarity_threshold = st.slider("유사도 임계값 (Tanimoto)", 0.5, 1.0, 0.8, 0.05)
    activity_diff_threshold = st.number_input("활성도 차이 임계값 (pKi)", min_value=0.1, value=1.0, step=0.1)

# --- 메인 페이지 ---
st.title("🧪 AI 기반 자동 약물 구조-활성 분석 시스템")
st.caption("AIGEN SCIENCES & 모두의연구소 PoC")

# 데이터 로드
df = None
if use_sample_data:
    df = load_data("data/sample_data.csv")
elif uploaded_file:
    df = load_data(uploaded_file)

if df is not None:
    st.subheader("입력 데이터 미리보기")
    st.dataframe(df.head())

    # 데이터 시각화 (Phase 1)
    st.subheader("데이터 분포 시각화")
    fig = px.histogram(df, x='activity', title='활성도(pKi) 분포', labels={'activity': 'pKi 값'})
    st.plotly_chart(fig, use_container_width=True)

    # 분석 시작 버튼
    if st.button("SAR 분석 시작", type="primary", use_container_width=True):
        with st.spinner("Activity Cliff를 분석 중입니다..."):
            # 핵심 패턴 추출 (Phase 2)
            cliffs = find_activity_cliffs(df, similarity_threshold, activity_diff_threshold)

        if not cliffs:
            st.warning("설정된 조건에 맞는 Activity Cliff를 찾을 수 없습니다. 임계값을 조정해보세요.")
        else:
            st.success(f"총 {len(cliffs)}개의 Activity Cliff를 찾았습니다. 가장 큰 활성도 차이를 보이는 쌍을 분석합니다.")
            
            # 가장 유의미한 cliff 선택
            top_cliff = cliffs[0]
            
            # 리포트 생성 (Phase 4)
            st.header("자동 생성 SAR 요약 리포트")
            st.markdown("---")

            st.subheader("핵심 분석: 주요 활성 변화 요인 (Key Activity Cliff)")

            col1, col2 = st.columns(2)
            
            mol1_info = top_cliff['mol_1']
            mol2_info = top_cliff['mol_2']

            with col1:
                st.info(f"**화합물 1: {mol1_info['ID']}**")
                image_url = draw_molecule(mol1_info['SMILES'])
                if image_url:
                    st.image(image_url, caption=f"pKi: {mol1_info['activity']:.2f}")

            with col2:
                st.info(f"**화합물 2: {mol2_info['ID']}**")
                image_url = draw_molecule(mol2_info['SMILES'])
                if image_url:
                    st.image(image_url, caption=f"pKi: {mol2_info['activity']:.2f}")

            st.metric(label="Tanimoto 유사도", value=f"{top_cliff['similarity']:.3f}")
            st.metric(label="활성도(pKi) 차이", value=f"{top_cliff['activity_diff']:.3f}")

            st.markdown("---")
            st.subheader("자동화된 해석 및 가설 (AI-Generated Hypothesis)")
            
            # LLM 기반 해석 및 가설 생성 (Phase 3)
            with st.spinner("AI가 화학적 가설을 생성 중입니다..."):
                hypothesis = generate_hypothesis(top_cliff)
            
            if hypothesis:
                st.markdown(hypothesis)

else:
    st.info("분석을 시작하려면 사이드바에서 CSV 파일을 업로드하거나 샘플 데이터를 사용하세요.")
