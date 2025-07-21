import streamlit as st
import pandas as pd
from rdkit import Chem
from utils_v2 import (
    load_data, 
    find_activity_cliffs, 
    generate_hypothesis, 
    draw_molecule, 
    load_pretrained_model, 
    smiles_to_descriptors, 
    load_feature_list,
    propose_and_predict_analogs
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
st.caption("AIGEN SCIENCES & 모두의연구소 PoC")

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
                # --- NEW: RAG 적용 ---
                with st.spinner("관련 문헌 검색 및 AI 가설 생성 중..."):
                    hypothesis, source_info = generate_hypothesis(selected_cliff)
                
                if hypothesis:
                    st.markdown("**AI-Generated Hypothesis:**")
                    st.info(hypothesis)
                    
                    if source_info:
                        with st.expander("📚 참고 문헌 정보 (RAG 근거)"):
                            st.markdown(f"**제목:** {source_info['title']}")
                            st.markdown(f"**PubMed 링크:** [{source_info['pmid']}]({source_info['link']})")
                            st.text_area("초록:", source_info['abstract'], height=200)

# --- QSAR 예측 탭 ---
with tab2:
    st.header("QSAR 예측: 신규 분자 활성 예측")
    
    # 사전 훈련된 모델과 피처 목록 로드
    model, model_message = load_pretrained_model("sar-analysis-app/data/qsar_model_final.joblib")
    feature_list = load_feature_list("sar-analysis-app/data/features.json")
    
    if model and feature_list:
        st.success(model_message)
        
        st.subheader("최적화할 기준 화합물 정보 입력")
        base_smiles = st.text_input("SMILES 문자열을 입력하세요:", "c1ccc(cc1)c2cnc3ccccc3c2")
        
        if st.button("AI 최적화 및 활성 예측", type="primary", key='qsar_button'):
            if base_smiles:
                features = smiles_to_descriptors(base_smiles, feature_list)
                if features is not None:
                    features_array = features.reshape(1, -1)
                    base_predicted_pki = model.predict(features_array)[0]
                    
                    st.subheader("🔬 분석 결과")
                    
                    # 기준 화합물 정보 표시
                    with st.container(border=True):
                        st.write("**기준 화합물**")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(draw_molecule(base_smiles))
                        with col2:
                            st.metric(label="예상 pKi", value=f"{base_predicted_pki:.3f}")
                            st.caption("이 화합물을 기반으로 AI가 더 좋은 구조를 제안합니다.")

                    st.markdown("---")
                    
                    # AI가 제안하는 신규 화합물
                    with st.spinner("AI가 활성도 개선을 위한 새로운 분자를 설계하고 있습니다..."):
                        proposals = propose_and_predict_analogs(base_smiles, model, feature_list)
                    
                    if proposals:
                        st.write("**AI 제안 신규 화합물**")
                        cols = st.columns(len(proposals))
                        for i, p in enumerate(proposals):
                            with cols[i]:
                                with st.container(border=True):
                                    st.image(draw_molecule(p['smiles']))
                                    delta_pki = p['predicted_pki'] - base_predicted_pki
                                    st.metric(label="예상 pKi", value=f"{p['predicted_pki']:.3f}", delta=f"{delta_pki:+.2f}")
                                    st.caption(f"**변경 이유:** {p['reason']}")
                    else:
                        st.warning("AI가 유효한 개선안을 제안하지 못했습니다. 다른 SMILES로 시도해보세요.")

                else:
                    st.error("유효하지 않은 SMILES 문자열입니다. 다시 확인해주세요.")
    else:
        if not model: st.error(model_message)
        if not feature_list: st.error("오류: 'features.json' 파일을 찾을 수 없습니다.")
