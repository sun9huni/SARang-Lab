import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import google.generativeai as genai
from urllib.parse import quote
import joblib

# --- Phase 1: 데이터 준비 및 탐색 ---

@st.cache_data
def load_data(uploaded_file):
    """업로드된 파일을 Pandas DataFrame으로 로드합니다."""
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str) and uploaded_file == "data/sample_data.csv":
                 df = pd.read_csv(uploaded_file, comment='#')
            else:
                 df = pd.read_csv(uploaded_file)

            if 'SMILES' not in df.columns or 'ID' not in df.columns or len(df.columns) < 3:
                st.error("CSV 파일은 'ID', 'SMILES', 그리고 활성도(activity) 컬럼을 포함해야 합니다.")
                return None
            df = df.rename(columns={df.columns[-1]: 'activity'})
            return df
        except Exception as e:
            st.error(f"데이터 로딩 중 오류 발생: {e}")
            return None
    return None

# --- QSAR 모델 로드 ---

def get_morgan_fingerprint(mol):
    """분자 객체로부터 Morgan Fingerprint를 생성합니다."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

@st.cache_resource # 모델 객체는 st.cache_resource를 사용
def load_pretrained_model(model_path="qsar_model.joblib"):
    """사전 훈련된 QSAR 모델 파일을 불러옵니다."""
    try:
        model = joblib.load(model_path)
        return model, "사전 훈련된 QSAR 모델을 성공적으로 불러왔습니다."
    except FileNotFoundError:
        return None, "오류: 'qsar_model.joblib' 파일을 찾을 수 없습니다. 모델 파일을 리포지토리에 업로드했는지 확인해주세요."
    except Exception as e:
        return None, f"모델 로딩 중 오류 발생: {e}"


# --- Phase 2: 핵심 패턴 자동 추출 (Activity Cliff) ---

@st.cache_data
def find_activity_cliffs(_df, similarity_threshold=0.8, activity_diff_threshold=1.0):
    """DataFrame에서 Activity Cliff 쌍을 찾습니다."""
    df = _df.copy()
    df.dropna(subset=['SMILES'], inplace=True)
    df['SMILES'] = df['SMILES'].astype(str)
    
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df.dropna(subset=['mol'], inplace=True)
    
    if df.empty:
        return []

    df['fp'] = df['mol'].apply(get_morgan_fingerprint)
    
    cliffs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            fp1 = df['fp'].iloc[i]
            fp2 = df['fp'].iloc[j]
            
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            if similarity >= similarity_threshold:
                activity1 = df['activity'].iloc[i]
                activity2 = df['activity'].iloc[j]
                activity_diff = abs(activity1 - activity2)
                
                if activity_diff >= activity_diff_threshold:
                    cliffs.append({
                        'mol_1': df.iloc[i],
                        'mol_2': df.iloc[j],
                        'similarity': similarity,
                        'activity_diff': activity_diff
                    })
    
    cliffs.sort(key=lambda x: x['activity_diff'], reverse=True)
    return cliffs

# --- Phase 3: LLM 기반 해석 및 가설 생성 ---

def generate_hypothesis(cliff):
    """Gemini API를 사용하여 Activity Cliff에 대한 화학적 가설을 생성합니다."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception:
        st.error("Gemini API 키를 찾을 수 없습니다. Streamlit secrets에 키를 설정해주세요.")
        return None

    model = genai.GenerativeModel('gemini-2.0-flash')

    mol1_info = cliff['mol_1']
    mol2_info = cliff['mol_2']
    compound_a = mol1_info if mol1_info['activity'] < mol2_info['activity'] else mol2_info
    compound_b = mol1_info if mol1_info['activity'] > mol2_info['activity'] else mol1_info

    prompt = f"""
        당신은 숙련된 신약 개발 화학자입니다. 두 화합물의 구조-활성 관계(SAR)에 대한 분석을 요청받았습니다.

        **분석 대상:**
        - **화합물 A (낮은 활성):**
          - ID: {compound_a['ID']}
          - SMILES: {compound_a['SMILES']}
          - 활성도 (pKi): {compound_a['activity']:.2f}
        - **화합물 B (높은 활성):**
          - ID: {compound_b['ID']}
          - SMILES: {compound_b['SMILES']}
          - 활성도 (pKi): {compound_b['activity']:.2f}

        **분석 요청:**
        두 화합물은 구조적으로 매우 유사하지만(Tanimoto 유사도: {cliff['similarity']:.2f}), 활성도에서 큰 차이(pKi 차이: {cliff['activity_diff']:.2f})를 보입니다.
        이러한 'Activity Cliff' 현상을 유발하는 핵심적인 구조적 차이점을 찾아내고, 그 차이가 어떻게 활성도 증가로 이어졌는지에 대한 화학적 가설을 전문가의 관점에서 설명해주세요.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API 호출 중 오류 발생: {e}")
        return "가설 생성에 실패했습니다."


# --- Phase 4: 리포트 생성 (시각화) ---

def draw_molecule(smiles_string):
    """SMILES 문자열로부터 NCI CACTUS 웹 서비스를 통해 분자 구조 이미지 URL을 생성합니다."""
    encoded_smiles = quote(smiles_string)
    return f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/image?format=png&width=350&height=350"
