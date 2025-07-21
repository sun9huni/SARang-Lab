import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import google.generativeai as genai
from urllib.parse import quote
import joblib
import json
import re

# --- Phase 1: 데이터 준비 및 탐색 ---
@st.cache_data
def load_data(uploaded_file):
    """업로드된 파일을 Pandas DataFrame으로 로드합니다."""
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str) and ("sample_data.csv" in uploaded_file or "large_sar_data.csv" in uploaded_file):
                 df = pd.read_csv(uploaded_file, comment='#')
            else:
                 df = pd.read_csv(uploaded_file)
            if 'SMILES' not in df.columns or 'ID' not in df.columns or len(df.columns) < 3:
                st.error("CSV 파일은 'ID', 'SMILES', 그리고 활성도(activity) 컬럼을 포함해야 합니다.")
                return None
            df = df.rename(columns={df.columns[2]: 'activity'})
            
            # 입체화학 정보를 포함하여 Canonical SMILES로 변환하는 함수
            def canonicalize_smiles(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                return None
            
            df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)
            df.dropna(subset=['SMILES'], inplace=True)
            
            return df
        except Exception as e:
            st.error(f"데이터 로딩 중 오류 발생: {e}")
            return None
    return None

# --- QSAR 모델 로드 및 예측용 피처 생성 ---

@st.cache_data
def load_feature_list(path="sar-analysis-app/data/features.json"):
    """훈련에 사용된 피처 목록을 불러옵니다."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    
@st.cache_resource
def load_pretrained_model(model_path="sar-analysis-app/data/qsar_model_final.joblib"):
    """사전 훈련된 QSAR 모델 파일을 불러옵니다."""
    try:
        model = joblib.load(model_path)
        return model, "사전 훈련된 QSAR 모델을 성공적으로 불러왔습니다."
    except FileNotFoundError:
        return None, f"오류: '{model_path}' 파일을 찾을 수 없습니다."
    except Exception as e:
        return None, f"모델 로딩 중 오류 발생: {e}"
# --- QSAR 예측 관련 함수 ---
def smiles_to_descriptors(smiles, feature_list):
    """(QSAR 예측용) SMILES로부터 훈련 시 사용된 피처 목록과 동일한 기술자 벡터를 생성합니다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 현재 환경에서 계산 가능한 모든 기술자를 계산
    all_descriptors = {name: func(mol) for name, func in Descriptors.descList}
    # 훈련 시 사용된 피처 목록 순서대로 값을 추출합니다.
    # 만약 현재 환경에서 계산할 수 없는 피처가 목록에 있다면, 0으로 처리합니다.
    descriptor_values = [all_descriptors.get(name, 0) for name in feature_list]
    return np.nan_to_num(np.array(descriptor_values), nan=0.0, posinf=0.0, neginf=0.0)

        
# --- Scaffold 기반 분석 함수 ---
def get_scaffold(smiles):
    """SMILES에서 Murcko Scaffold를 추출합니다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=True)
        except:
            return None
    return None

@st.cache_data
def find_scaffold_matches(_training_df, new_smiles):
    """훈련 데이터에서 동일한 Scaffold를 가진 화합물들의 활성도(pKi)를 찾습니다."""
    new_scaffold = get_scaffold(new_smiles)
    if not new_scaffold:
        return []
    
    if 'scaffold' not in _training_df.columns:
        _training_df['scaffold'] = _training_df['SMILES'].apply(get_scaffold)
    
    matches = _training_df[_training_df['scaffold'] == new_scaffold]
    return matches['activity'].tolist()

# --- SAR 분석용 함수들 ---
def get_morgan_fingerprint(mol):
    """(SAR 분석용) 분자 객체로부터 Morgan Fingerprint를 생성합니다."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

@st.cache_data
def prepare_comparison_data(_df):
    """유사도 비교를 위해 훈련 데이터의 분자 객체와 핑거프린트를 미리 계산합니다."""
    df = _df.copy()
    df.dropna(subset=['SMILES'], inplace=True)
    df['SMILES'] = df['SMILES'].astype(str)
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df.dropna(subset=['mol'], inplace=True)
    df['fp'] = df['mol'].apply(get_morgan_fingerprint)
    return df

def find_most_similar_compounds(new_smiles, training_df, top_n=2):
    """새로운 화합물과 가장 유사한 화합물을 훈련 데이터에서 찾습니다."""
    new_mol = Chem.MolFromSmiles(new_smiles)
    if not new_mol: return []
    new_fp = get_morgan_fingerprint(new_mol)
    training_df['similarity'] = training_df['fp'].apply(lambda x: DataStructs.TanimotoSimilarity(x, new_fp))
    most_similar = training_df.sort_values(by='similarity', ascending=False).head(top_n)
    return most_similar.to_dict('records')

def find_activity_cliffs(_df, similarity_threshold=0.8, activity_diff_threshold=1.0):
    df = _df.copy()
    df.dropna(subset=['SMILES'], inplace=True)
    df['SMILES'] = df['SMILES'].astype(str)
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df.dropna(subset=['mol'], inplace=True)
    if df.empty: return []
    df['fp'] = df['mol'].apply(get_morgan_fingerprint)
    cliffs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            fp1, fp2 = df['fp'].iloc[i], df['fp'].iloc[j]
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            if similarity >= similarity_threshold:
                activity1, activity2 = df['activity'].iloc[i], df['activity'].iloc[j]
                activity_diff = abs(activity1 - activity2)
                if activity_diff >= activity_diff_threshold:
                    cliffs.append({'mol_1': df.iloc[i], 'mol_2': df.iloc[j], 'similarity': similarity, 'activity_diff': activity_diff})
    cliffs.sort(key=lambda x: x['activity_diff'], reverse=True)
    return cliffs

def generate_hypothesis(cliff):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception:
        st.error("Gemini API 키를 찾을 수 없습니다.")
        return None
    model = genai.GenerativeModel('gemini-2.0-flash')
    mol1_info, mol2_info = cliff['mol_1'], cliff['mol_2']
    compound_a = mol1_info if mol1_info['activity'] < mol2_info['activity'] else mol2_info
    compound_b = mol1_info if mol1_info['activity'] > mol2_info['activity'] else mol1_info
    
    prompt_addition = ""
    mol_a = Chem.MolFromSmiles(compound_a['SMILES'])
    mol_b = Chem.MolFromSmiles(compound_b['SMILES'])
    if mol_a and mol_b:
        base_smiles_a = Chem.MolToSmiles(mol_a, isomericSmiles=False, canonical=True)
        base_smiles_b = Chem.MolToSmiles(mol_b, isomericSmiles=False, canonical=True)
        if base_smiles_a == base_smiles_b and compound_a['SMILES'] != compound_b['SMILES']:
            prompt_addition = "특히, 이 두 화합물은 동일한 2D 구조를 가진 입체이성질체(stereoisomer)입니다. 3D 공간 배열의 차이가 어떻게 이러한 활성 차이를 유발하는지 집중적으로 설명해주세요."

    prompt = f"당신은 숙련된 신약 개발 화학자입니다. 두 화합물의 구조-활성 관계(SAR)에 대한 분석을 요청받았습니다.\n\n**분석 대상:**\n- **화합물 A (낮은 활성):**\n  - ID: {compound_a['ID']}\n  - SMILES: {compound_a['SMILES']}\n  - 활성도 (pKi): {compound_a['activity']:.2f}\n- **화합물 B (높은 활성):**\n  - ID: {compound_b['ID']}\n  - SMILES: {compound_b['SMILES']}\n  - 활성도 (pKi): {compound_b['activity']:.2f}\n\n**분석 요청:**\n두 화합물은 구조적으로 매우 유사하지만(Tanimoto 유사도: {cliff['similarity']:.2f}), 활성도에서 큰 차이(pKi 차이: {cliff['activity_diff']:.2f})를 보입니다.\n이러한 'Activity Cliff' 현상을 유발하는 핵심적인 구조적 차이점을 찾아내고, 그 차이가 어떻게 활성도 증가로 이어졌는지에 대한 화학적 가설을 전문가의 관점에서 설명해주세요. {prompt_addition}"
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API 호출 중 오류 발생: {e}")
        return "가설 생성에 실패했습니다."

def draw_molecule(smiles_string):
    encoded_smiles = quote(smiles_string)
    return f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/image?format=png&width=350&height=350"

# --- NEW: AI 기반 분자 제안 및 예측 함수 ---
def propose_and_predict_analogs(base_smiles, qsar_model, feature_list):
    """AI를 통해 개선된 분자 구조를 제안받고, QSAR 모델로 활성을 예측합니다."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception:
        st.error("Gemini API 키를 찾을 수 없습니다.")
        return []

    model_gen = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
        당신은 신약 개발 전문가인 숙련된 의약화학자입니다.

        **과제:**
        주어진 기준 화합물(base compound)의 구조를 분석하고, 활성도(potency)를 향상시킬 수 있는 새로운 유사체(analog) 3개를 제안해주세요. 제안하는 분자는 기존 구조와 유사해야 하며, 합성이 가능한 현실적인 구조여야 합니다.

        **기준 화합물 SMILES:**
        {base_smiles}

        **지침:**
        1.  기준 화합물의 SAR(구조-활성 관계)을 고려하여, 활성을 높일 수 있는 일반적인 의약화학 전략을 적용하세요. (예: 수소 결합 주개/받개 추가, 소수성 포켓 상호작용 강화, 고리 시스템 변경 등)
        2.  제안하는 각 유사체에 대해, 변경된 부분과 예상되는 활성 향상 이유를 한 문장으로 간략하게 설명해주세요.
        3.  결과는 반드시 아래의 형식에 맞춰, 각 줄에 "SMILES,이유" 형태로 작성해주세요. 다른 설명은 추가하지 마세요.

        **출력 형식 예시:**
        CCOc1ccccc1,알코올을 에틸 에터로 변경하여 소수성을 증가시킴.
        c1ccc(C(F)(F)F)cc1,벤젠 고리에 강력한 전자 끌기 그룹을 추가하여 상호작용을 변경함.
    """

    try:
        response = model_gen.generate_content(prompt)
        
        proposals = []
        # AI 응답에서 SMILES 코드와 설명을 추출
        lines = response.text.strip().split('\n')
        for line in lines:
            parts = line.split(',')
            if len(parts) == 2:
                smiles = parts[0].strip()
                reason = parts[1].strip()
                mol = Chem.MolFromSmiles(smiles)
                if mol: # 유효한 SMILES인지 확인
                    features = smiles_to_descriptors(smiles, feature_list)
                    if features is not None:
                        features_array = features.reshape(1, -1)
                        predicted_pki = qsar_model.predict(features_array)[0]
                        proposals.append({
                            "smiles": smiles,
                            "reason": reason,
                            "predicted_pki": predicted_pki
                        })
        return proposals

    except Exception as e:
        st.error(f"AI 분자 제안 중 오류 발생: {e}")
        return []
