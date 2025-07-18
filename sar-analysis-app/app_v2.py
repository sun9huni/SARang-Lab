import streamlit as st
import pandas as pd
from rdkit import Chem
from utils_v2 import (
    load_data, find_activity_cliffs, generate_hypothesis, draw_molecule, 
    load_pretrained_model, get_morgan_fingerprint, prepare_comparison_data, 
    find_most_similar_compounds, smiles_to_descriptors, load_feature_list
)
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="AI 기반 SAR/QSAR 분석 시스템 v2", page_icon="💊", layout="wide")

# --- 사이드바 ---
with st.sidebar:
    st.title("분석 설정 (v2)")
    st.markdown("---")
    uploaded_file = st.file_uploader("SAR 데이터(.csv)를 업로드하세요.", type="csv")
    use_sample_data = st.checkbox("샘플 데이터 사용", value=True)
    st.markdown("---")
    st.subheader("SAR 분석 파라미터")
    similarity_threshold = st.slider("유사도 임계값 (Tanimoto)", 0.5, 1.0, 0.8, 0.05)
    activity_diff_threshold = st.number_input("활성도 차이 임계값 (pKi)", min_value=0.1, value=1.0, step=0.1)

# --- 메인 페이지 ---
st.title("💊 AI 기반 SAR & QSAR 분석 시스템 (v2)")
st.caption("AIGEN SCIENCES & 모두의연구소 PoC - 사전 학습 모델 적용")

df = None
if use_sample_data:
    df = load_data("sar-analysis-app/data/large_sar_data.csv")
elif uploaded_file:
    df = load_data(uploaded_file)

tab1, tab2 = st.tabs(["SAR 분석 (Activity Cliff)", "QSAR 예측 (신규 분자)"])

# --- SAR 분석 탭 ---
with tab1:
    st.header("SAR 분석: 주요 활성 변화 요인")
    if df is not None:
        st.dataframe(df.head())
        fig = px.histogram(df, x='activity', title='활성도(pKi) 분포', labels={'activity': 'pKi 값'})
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Activity Cliff 분석 시작", type="primary", key='sar_button'):
            with st.spinner("Activity Cliff를 분석 중입니다..."):
                cliffs = find_activity_cliffs(df, similarity_threshold, activity_diff_threshold)
            st.session_state['cliffs'] = cliffs
        if 'cliffs' in st.session_state:
            cliffs = st.session_state['cliffs']
            if not cliffs:
                st.warning("설정된 조건에 맞는 Activity Cliff를 찾을 수 없습니다.")
            else:
                st.success(f"총 {len(cliffs)}개의 Activity Cliff를 찾았습니다. 분석할 쌍을 선택하세요.")
                cliff_options = [f"{i+1}. {c['mol_1']['ID']} vs {c['mol_2']['ID']} (ΔpKi: {c['activity_diff']:.2f})" for i, c in enumerate(cliffs)]
                selected_option = st.selectbox("분석할 Activity Cliff 선택:", cliff_options, key='cliff_select')
                selected_index = cliff_options.index(selected_option)
                selected_cliff = cliffs[selected_index]
                st.subheader("SAR 요약 리포트")
                col1, col2 = st.columns(2)
                mol1_info, mol2_info = selected_cliff['mol_1'], selected_cliff['mol_2']
                with col1:
                    st.info(f"**화합물 1: {mol1_info['ID']}**")
                    st.image(draw_molecule(mol1_info['SMILES']), caption=f"pKi: {mol1_info['activity']:.2f}")
                with col2:
                    st.info(f"**화합물 2: {mol2_info['ID']}**")
                    st.image(draw_molecule(mol2_info['SMILES']), caption=f"pKi: {mol2_info['activity']:.2f}")
                st.metric("Tanimoto 유사도", f"{selected_cliff['similarity']:.3f}")
                with st.spinner("AI가 화학적 가설을 생성 중입니다..."):
                    hypothesis = generate_hypothesis(selected_cliff)
                if hypothesis:
                    st.markdown("**AI-Generated Hypothesis:**")
                    st.info(hypothesis)
    else:
        st.info("SAR 분석을 시작하려면 사이드바에서 CSV 파일을 업로드하거나 샘플 데이터를 사용하세요.")

# --- QSAR 예측 탭 ---
with tab2:
    st.header("QSAR 예측: 신규 분자 활성 예측")
    
    # 사전 훈련된 모델과 피처 목록 로드
    model, model_message = load_pretrained_model("sar-analysis-app/data/qsar_model_final.joblib")
    feature_list = load_feature_list("sar-analysis-app/data/features.json")
    
    if model and feature_list:
        st.success(model_message)
        training_data = load_data("sar-analysis-app/data/large_sar_data.csv")
        if training_data is not None:
            comparison_df = prepare_comparison_data(training_data)
            high_potency_threshold = training_data['activity'].quantile(0.75)
            low_potency_threshold = training_data['activity'].quantile(0.25)

            st.subheader("신규 화합물 정보 입력")
            new_smiles = st.text_input("활성을 예측할 분자의 SMILES 문자열을 입력하세요:", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC")
            
            if st.button("활성 예측", type="primary", key='qsar_button'):
                if new_smiles:
                    # --- FIX: 예측 파이프라인 수정 ---
                    # 1. SMILES로부터 훈련과 동일한 피처 목록으로 기술자 계산
                    features = smiles_to_descriptors(new_smiles, feature_list)
                    
                    if features is not None:
                        features_array = features.reshape(1, -1)
                        
                        # 2. 불러온 모델로 바로 예측 수행
                        predicted_activity = model.predict(features_array)[0]
                        
                        st.subheader("📈 예측 결과 분석")
                        
                        if predicted_activity >= high_potency_threshold:
                            grade = "High Potency"
                            st.success(f"**등급: {grade} (상위 25% 이상)**")
                        elif predicted_activity <= low_potency_threshold:
                            grade = "Low Potency"
                            st.error(f"**등급: {grade} (하위 25% 이하)**")
                        else:
                            grade = "Medium Potency"
                            st.info(f"**등급: {grade}**")
                        st.metric(label="예측된 pKi 활성도", value=f"{predicted_activity:.3f}")
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=training_data['activity'], name='훈련 데이터 분포', marker_color='#3b82f6'))
                        fig.add_vline(x=predicted_activity, line_width=3, line_dash="dash", line_color="red",
                                      annotation_text=f"예측값: {predicted_activity:.2f}", 
                                      annotation_position="top right")
                        fig.update_layout(title_text='훈련 데이터 활성도 분포 및 예측값 위치', xaxis_title='pKi 값', yaxis_title='빈도')
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("🔬 유사 화합물 비교 (훈련 데이터 기준)")
                        with st.spinner("유사 화합물을 검색 중입니다..."):
                            similar_compounds = find_most_similar_compounds(new_smiles, comparison_df)
                        if similar_compounds:
                            cols = st.columns(len(similar_compounds))
                            for i, comp in enumerate(similar_compounds):
                                with cols[i]:
                                    st.info(f"**Top {i+1} 유사 화합물**")
                                    st.image(draw_molecule(comp['SMILES']), caption=f"ID: {comp['ID']}")
                                    st.metric(label="실제 pKi", value=f"{comp['activity']:.3f}")
                                    st.metric(label="유사도", value=f"{comp['similarity']:.3f}")
                        else:
                            st.warning("훈련 데이터에서 유사한 화합물을 찾을 수 없습니다.")
                    else:
                        st.error("유효하지 않은 SMILES 문자열입니다. 다시 확인해주세요.")
    else:
        if not model: st.error(model_message)
        if not feature_list: st.error("오류: 'features.json' 파일을 찾을 수 없습니다.")
