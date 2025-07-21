import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
import google.generativeai as genai
from urllib.parse import quote
import joblib
import json

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
