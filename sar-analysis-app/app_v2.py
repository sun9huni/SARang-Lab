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
    st.title("🔬 AI-SAR/QSAR 분석 시스템")
    st.markdown("---")

    # 1. 타겟 단백질 입력 (RAG 검색에 사용)
    target_name = st.text_input(
        "**1. 분석 대상 타겟 단백질**", 
        "EGFR", 
        help="AI 가설 생성 시, 이 타겟에 대한 최신 논문을 참조하여 신뢰도를 높입니다. 예: EGFR, CDK2, HSP90"
    ).strip().upper()

    st.markdown("---")

    # 2. 데이터 소스 선택
    st.markdown("**2. SAR 분석용 데이터**")
    source_selection = st.radio(
        "데이터 소스 선택", 
        ('샘플 데이터 사용', '파일 업로드'), 
        key='source_select', 
        label_visibility="collapsed"
    )
    uploaded_file = None
    if source_selection == '파일 업로드':
        uploaded_file = st.file_uploader("SAR 데이터(.csv)를 업로드하세요.", type=['csv'])
    else:
        uploaded_file = "sar-analysis-app/data/large_sar_data.csv"
        
    st.markdown("---")
    st.info("타겟을 지정하면, SAR 리포트의 AI 가설이 해당 타겟에 맞춰 자동으로 최적화됩니다.")

df = load_data(uploaded_file)
# QSAR은 타겟과 무관하게 단일 모델을 사용하므로, 훈련 데이터 로딩은 제거

# --- 메인 페이지 ---
if df is not None:
    tab1, tab2 = st.tabs(["SAR 분석 (Activity Cliff)", "QSAR 예측 (AI 분자 최적화)"])

    # ==================================
    # SAR 분석 탭 (타겟-특화 RAG 적용)
    # ==================================
    with tab1:
        st.header(f"🎯 {target_name or '범용'} 타겟 Activity Cliff 분석 리포트")
        st.markdown("`Activity Cliff`란, 구조는 매우 유사하지만 활성도에서 큰 차이를 보이는 화합물 쌍을 의미합니다. 이는 신약 개발의 중요한 단서를 제공합니다.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("분석 조건 설정")
            similarity_threshold = st.slider('유사도 임계값 (Tanimoto)', 0.5, 1.0, 0.8, 0.01)
            activity_diff_threshold = st.slider('활성도 차이 임계값 (pKi)', 0.5, 3.0, 1.0, 0.1)

        with col2:
            st.subheader("분석 대상 데이터")
            st.dataframe(df, height=200)

        st.markdown("---")
        
        if st.button("SAR 분석 시작", type="primary", use_container_width=True):
            if not target_name:
                st.error("사이드바에서 분석 대상 타겟 단백질 이름을 먼저 입력해주세요.")
            else:
                cliffs = find_activity_cliffs(df, similarity_threshold, activity_diff_threshold)
                
                if not cliffs:
                    st.warning("설정된 조건에 맞는 Activity Cliff를 찾을 수 없습니다. 임계값을 조정해보세요.")
                else:
                    selected_cliff = cliffs[0]
                    mol1, mol2 = selected_cliff['mol_1'], selected_cliff['mol_2']
                    
                    st.subheader("📊 핵심 분석: 주요 활성 변화 요인 (Key Activity Cliff)")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(draw_molecule(mol1['SMILES']), caption=f"화합물 1: {mol1['ID']}")
                        st.metric("pKi", f"{mol1['activity']:.2f}")
                    with c2:
                        st.image(draw_molecule(mol2['SMILES']), caption=f"화합물 2: {mol2['ID']}")
                        st.metric("pKi", f"{mol2['activity']:.2f}")
                    
                    st.metric("Tanimoto 유사도", f"{selected_cliff['similarity']:.3f}", 
                              delta=f"활성도(pKi) 차이: {selected_cliff['activity_diff']:.3f}", 
                              delta_color="off")
                    
                    with st.spinner(f"{target_name} 관련 문헌을 참조하여 가설을 생성 중입니다..."):
                        hypothesis, source_info = generate_hypothesis(selected_cliff, target_name)
                    
                    st.markdown("##### AI-Generated Hypothesis:")
                    st.markdown(hypothesis)

                    if source_info:
                        with st.expander("📚 참고 문헌 정보 (RAG 근거)"):
                            st.markdown(f"**제목:** {source_info['title']}")
                            st.markdown(f"**링크:** [PubMed 바로가기]({source_info['link']})")
                            st.caption(f"**초록:** {source_info['abstract']}")

    # ==================================
    # QSAR 예측 탭 (단일 모델 사용)
    # ==================================
    with tab2:
        st.header("💡 AI 기반 분자 최적화 제안")
        st.markdown("기준 화합물의 SMILES를 입력하면, AI가 활성도 개선이 예상되는 새로운 분자 구조를 제안하고, 사전 훈련된 QSAR 모델로 활성도를 예측합니다.")
        
        # 타겟과 상관없이 고정된 단일 모델과 피처 목록 로드
        model_pipeline, msg = load_pretrained_model()
        features, f_msg = load_feature_list()

        if model_pipeline and features:
            base_smiles = st.text_input("기준 화합물 SMILES 입력", "c1ccc(cc1)c2[nH]c3ccc(C)cc3n2")

            if st.button("AI 최적화 제안 받기", type="primary", use_container_width=True):
                base_mol = Chem.MolFromSmiles(base_smiles)
                if base_mol:
                    with st.spinner("AI가 새로운 분자를 설계하고 QSAR 모델로 활성을 예측 중입니다..."):
                        proposals = propose_and_predict_analogs(base_smiles, model_pipeline, features)
                    
                    if proposals:
                        st.subheader("✨ AI 제안 및 예측 결과")
                        
                        st.markdown("---")
                        st.markdown(f"**기준 화합물:** `{base_smiles}`")
                        base_features = smiles_to_descriptors(base_smiles, features)
                        if base_features is not None:
                            feature_df = pd.DataFrame([base_features], columns=features)
                            predicted_base_pki = model_pipeline.predict(feature_df)[0]
                            st.metric("기준 화합물 예측 pKi", f"{predicted_base_pki:.2f}")
                        st.image(draw_molecule(base_smiles))
                        st.markdown("---")

                        for i, prop in enumerate(proposals):
                            st.markdown(f"##### 제안 {i+1}")
                            st.image(draw_molecule(prop['smiles']))
                            st.metric(f"제안 {i+1} 예측 pKi", f"{prop['predicted_pki']:.2f}", delta=f"{prop['predicted_pki'] - predicted_base_pki:.2f}")
                            st.info(f"**AI 제안 이유:** {prop['reason']}")
                            st.code(prop['smiles'], language='text')
                            st.markdown("---")

                    else:
                        st.error("AI가 유효한 분자를 제안하지 못했습니다. 잠시 후 다시 시도해주세요.")
                else:
                    st.error("입력한 SMILES 문자열이 유효하지 않습니다.")
        else:
            st.error(f"모델 또는 피처 목록을 불러오는 데 실패했습니다: {msg or f_msg}")

else:
    st.warning("데이터를 불러오지 못했습니다. 사이드바에서 파일을 업로드하거나 샘플 데이터를 선택해주세요.")
