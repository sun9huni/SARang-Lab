import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from openai import OpenAI
from urllib.parse import quote
import joblib
import json
import requests
import xml.etree.ElementTree as ET

# --- 데이터 처리 및 모델 로드 (기존과 동일) ---
@st.cache_data
def load_data(uploaded_file):
    """업로드된 파일을 Pandas DataFrame으로 로드하고 SMILES를 표준화합니다."""
    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str) and ("sample_data.csv" in uploaded_file or "large_sar_data.csv" in uploaded_file):
                 df = pd.read_csv(uploaded_file, comment='#')
            else:
                 df = pd.read_csv(uploaded_file)
            if 'SMILES' not in df.columns or 'ID' not in df.columns or len(df.columns) < 3:
                st.error("CSV 파일은 'ID', 'SMILES', 그리고 활성도(activity) 컬럼을 포함해야 합니다.")
                return None
            df = df.rename(columns={df.columns[-1]: 'activity'})
            def canonicalize_smiles(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol: return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                return None
            df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)
            df.dropna(subset=['SMILES'], inplace=True)
            return df
        except Exception as e:
            st.error(f"데이터 로딩 중 오류 발생: {e}")
    return None

# --- RAG 및 기타 유틸리티 함수 (기존과 동일) ---
def get_structural_difference_keyword(smiles1, smiles2):
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if not mol1 or not mol2: return None
    mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=5)
    if mcs_result.numAtoms == 0: return None
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    diff_mol = Chem.DeleteSubstructs(mol1 if mol1.GetNumAtoms() > mol2.GetNumAtoms() else mol2, mcs_mol)
    if diff_mol.GetNumAtoms() == 0: return None
    try: return Chem.rdMolDescriptors.CalcMolFormula(diff_mol)
    except: return None

@st.cache_data
def search_pubmed_for_context(smiles1, smiles2, target_name="EGFR", max_results=1):
    def fetch_articles(search_term):
        try:
            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            esearch_params = {'db': 'pubmed', 'term': search_term, 'retmax': max_results, 'sort': 'relevance'}
            esearch_response = requests.get(esearch_url, params=esearch_params, timeout=10)
            esearch_response.raise_for_status()
            es_root = ET.fromstring(esearch_response.content)
            id_list = [elem.text for elem in es_root.findall('.//Id')]
            if not id_list: return None

            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            efetch_params = {'db': 'pubmed', 'id': ",".join(id_list), 'retmode': 'xml'}
            efetch_response = requests.get(efetch_url, params=efetch_params, timeout=10)
            efetch_response.raise_for_status()
            ef_root = ET.fromstring(efetch_response.content)
            article = ef_root.find('.//PubmedArticle')
            if article:
                title = article.findtext('.//ArticleTitle', 'No title found')
                abstract = article.findtext('.//Abstract/AbstractText', 'No abstract found')
                pmid = article.findtext('.//PMID', '')
                return {"title": title, "abstract": abstract, "pmid": pmid, "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
        except Exception: return None
        return None
    diff_keyword = get_structural_difference_keyword(smiles1, smiles2)
    if diff_keyword and (result := fetch_articles(f'("{target_name}"[Title/Abstract]) AND ("{diff_keyword}"[Title/Abstract])')):
        return result
    return fetch_articles(f'("{target_name}"[Title/Abstract]) AND ("structure activity relationship"[Title/Abstract])')

# --- LLM 기반 가설 생성 (OpenAI 적용) ---
def generate_hypothesis(cliff):
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        st.error("OpenAI API 키를 찾을 수 없습니다. Streamlit secrets에 키를 설정해주세요.")
        return None, None

    model = "gpt-4o"
    compound_a, compound_b = (cliff['mol_1'], cliff['mol_2']) if cliff['mol_1']['activity'] < cliff['mol_2']['activity'] else (cliff['mol_2'], cliff['mol_1'])
    context_info = search_pubmed_for_context(compound_a['SMILES'], compound_b['SMILES'])

    rag_prompt_addition = f"**참고 문헌 정보:**\n- 제목: {context_info['title']}\n- 초록: {context_info['abstract']}\n\n위 참고 문헌의 내용을 바탕으로 가설을 생성해주세요." if context_info else ""
    prompt_addition = "특히, 이 두 화합물은 동일한 2D 구조를 가진 입체이성질체(stereoisomer)입니다. 3D 공간 배열의 차이가 어떻게 이러한 활성 차이를 유발하는지 집중적으로 설명해주세요." if Chem.MolToSmiles(Chem.MolFromSmiles(compound_a['SMILES']), isomericSmiles=False) == Chem.MolToSmiles(Chem.MolFromSmiles(compound_b['SMILES']), isomericSmiles=False) and compound_a['SMILES'] != compound_b['SMILES'] else ""

    system_prompt = "당신은 숙련된 신약 개발 화학자입니다. 두 화합물의 구조-활성 관계(SAR)에 대한 분석을 요청받았습니다. 분석 결과를 전문가의 관점에서 명확하고 간결하게 마크다운 형식으로 작성해주세요."
    user_prompt = f"**분석 대상:**\n- **화합물 A (낮은 활성):**\n  - ID: {compound_a['ID']}\n  - SMILES: {compound_a['SMILES']}\n  - 활성도 (pKi): {compound_a['activity']:.2f}\n- **화합물 B (높은 활성):**\n  - ID: {compound_b['ID']}\n  - SMILES: {compound_b['SMILES']}\n  - 활성도 (pKi): {compound_b['activity']:.2f}\n\n**분석 요청:**\n두 화합물은 구조적으로 매우 유사하지만(Tanimoto 유사도: {cliff['similarity']:.2f}), 활성도에서 큰 차이(pKi 차이: {cliff['activity_diff']:.2f})를 보입니다. 이러한 'Activity Cliff' 현상을 유발하는 핵심적인 구조적 차이점을 찾아내고, 그 차이가 어떻게 활성도 증가로 이어졌는지에 대한 화학적 가설을 설명해주세요. {prompt_addition}\n\n{rag_prompt_addition}"

    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        return response.choices[0].message.content, context_info
    except Exception as e:
        st.error(f"OpenAI API 호출 중 오류 발생: {e}")
        return "가설 생성에 실패했습니다.", None

# --- AI 기반 분자 최적화 제안 (OpenAI 적용) ---
def propose_and_predict_analogs(base_smiles, qsar_model, feature_list):
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        st.error("OpenAI API 키를 찾을 수 없습니다.")
        return []

    system_prompt = "당신은 신약 개발 전문가인 숙련된 의약화학자입니다. 결과를 반드시 지시된 형식에 맞춰, 각 줄에 \"SMILES,이유\" 형태로만 작성해주세요. 다른 설명은 절대로 추가하지 마세요."
    user_prompt = f"**과제:**\n주어진 기준 화합물(base compound)의 구조를 분석하고, 활성도(potency)를 향상시킬 수 있는 새로운 유사체(analog) 3개를 제안해주세요. 제안하는 분자는 기존 구조와 유사해야 하며, 합성이 가능한 현실적인 구조여야 합니다.\n\n**기준 화합물 SMILES:**\n{base_smiles}\n\n**지침:**\n1. 기준 화합물의 SAR(구조-활성 관계)을 고려하여, 활성을 높일 수 있는 일반적인 의약화학 전략을 적용하세요.\n2. 제안하는 각 유사체에 대해, 변경된 부분과 예상되는 활성 향상 이유를 한 문장으로 간략하게 설명해주세요."

    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        proposals_text = response.choices[0].message.content
        proposals = []
        for line in proposals_text.strip().split('\n'):
            if len(parts := line.split(',')) == 2:
                smiles, reason = parts[0].strip(), parts[1].strip()
                if (mol := Chem.MolFromSmiles(smiles)) and (features := smiles_to_descriptors(smiles, feature_list)) is not None:
                    predicted_pki = qsar_model.predict(features.reshape(1, -1))[0]
                    proposals.append({"smiles": smiles, "reason": reason, "predicted_pki": predicted_pki})
        return proposals
    except Exception as e:
        st.error(f"AI 분자 제안 중 오류 발생: {e}")
        return []

# --- 나머지 유틸리티 함수 (기존과 동일) ---
@st.cache_data
def load_feature_list(path="data/features.json"):
    try:
        with open(path, 'r') as f: return json.load(f)
    except FileNotFoundError: return None

@st.cache_resource
def load_pretrained_model(model_path="data/qsar_model_final.joblib"):
    try:
        return joblib.load(model_path), "사전 훈련된 QSAR 모델을 성공적으로 불러왔습니다."
    except Exception as e: return None, f"모델 로딩 중 오류 발생: {e}"

def smiles_to_descriptors(smiles, feature_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    all_descriptors = {name: func(mol) for name, func in Descriptors.descList}
    descriptor_values = [all_descriptors.get(name, 0) for name in feature_list]
    return np.nan_to_num(np.array(descriptor_values), nan=0.0, posinf=0.0, neginf=0.0)

def draw_molecule(smiles_string):
    return f"https://cactus.nci.nih.gov/chemical/structure/{quote(smiles_string)}/image?format=png&width=350&height=350"
