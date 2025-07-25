import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
import google.generativeai as genai
from urllib.parse import quote
import joblib
import json
import re
import requests
import xml.etree.ElementTree as ET

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

# --- RAG를 위한 PubMed 검색 기능 (개선) ---
def get_structural_difference_keyword(smiles1, smiles2):
    """두 SMILES의 구조적 차이를 나타내는 키워드를 찾습니다."""
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if not mol1 or not mol2: return None

    # 최대 공통 부분구조(MCS) 찾기
    mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=5, completeRingsOnly=True, ringMatchesRingOnly=True)
    if mcs_result.numAtoms == 0: return None
    
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    # 더 큰 분자에서 공통 구조를 제거하여 차이점을 찾음
    bigger_mol, smaller_mol = (mol1, mol2) if mol1.GetNumAtoms() > mol2.GetNumAtoms() else (mol2, mol1)
    
    try:
        fragments = Chem.DeleteSubstructs(bigger_mol, mcs_mol, onlyFrags=True)
        # 가장 큰 fragment의 분자식을 반환
        if frags := Chem.GetMolFrags(fragments, asMols=True):
            largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
            return Chem.rdMolDescriptors.CalcMolFormula(largest_frag)
    except Exception:
        return None # 오류 발생 시 None 반환
    return None

@st.cache_data
def search_pubmed_for_context(smiles1, smiles2, target_name, max_results=1):
    """PubMed에서 관련 문헌을 검색하여 컨텍스트를 제공합니다."""
    def fetch_articles(search_term):
        try:
            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {'db': 'pubmed', 'term': search_term, 'retmax': max_results, 'sort': 'relevance'}
            response = requests.get(esearch_url, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            id_list = [elem.text for elem in root.findall('.//Id')]
            if not id_list: return None

            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {'db': 'pubmed', 'id': ",".join(id_list), 'retmode': 'xml'}
            response = requests.get(efetch_url, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            article = root.find('.//PubmedArticle')
            if article:
                title = article.findtext('.//ArticleTitle', 'No title found')
                abstract = " ".join([p.text for p in article.findall('.//Abstract/AbstractText') if p.text])
                pmid = article.findtext('.//PMID', '')
                if not abstract: abstract = 'No abstract found'
                return {"title": title, "abstract": abstract, "pmid": pmid, "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
        except Exception:
            return None
        return None

    # 1차 검색: 구조적 차이점과 타겟으로 정밀 검색
    diff_keyword = get_structural_difference_keyword(smiles1, smiles2)
    if diff_keyword and (result := fetch_articles(f'("{target_name}"[Title/Abstract]) AND ("{diff_keyword}"[Title/Abstract])')):
        return result
    
    # 2차 검색: 결과가 없으면 일반적인 키워드로 검색
    return fetch_articles(f'("{target_name}"[Title/Abstract]) AND ("structure activity relationship"[Title/Abstract])')


# --- LLM 기반 가설 생성 (RAG 적용) ---
def generate_hypothesis(cliff, target_name, api_key, llm_provider):
    """선택된 LLM API를 사용하여 Activity Cliff에 대한 화학적 가설을 생성합니다."""
    if not api_key:
        st.error(f"사이드바에 {llm_provider} API 키를 입력해주세요.")
        return "API 키가 필요합니다.", None

    compound_a, compound_b = (cliff['mol_1'], cliff['mol_2']) if cliff['mol_1']['activity'] < cliff['mol_2']['activity'] else (cliff['mol_2'], cliff['mol_1'])
    context_info = search_pubmed_for_context(compound_a['SMILES'], compound_b['SMILES'], target_name)
    rag_prompt_addition = f"\n\n**참고 문헌 정보:**\n- 제목: {context_info['title']}\n- 초록: {context_info['abstract']}\n\n위 참고 문헌의 내용을 바탕으로 가설을 생성해주세요." if context_info else ""
    is_stereoisomer = (Chem.MolToSmiles(Chem.MolFromSmiles(compound_a['SMILES']), isomericSmiles=False) == Chem.MolToSmiles(Chem.MolFromSmiles(compound_b['SMILES']), isomericSmiles=False)) and (compound_a['SMILES'] != compound_b['SMILES'])
    prompt_addition = "\n\n특히, 이 두 화합물은 동일한 2D 구조를 가진 입체이성질체(stereoisomer)입니다. 3D 공간 배열의 차이가 어떻게 이러한 활성 차이를 유발하는지 집중적으로 설명해주세요." if is_stereoisomer else ""
    user_prompt = f"""
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
    이러한 'Activity Cliff' 현상을 유발하는 핵심적인 구조적 차이점을 찾아내고, 그 차이가 어떻게 활성도 증가로 이어졌는지에 대한 화학적 가설을 설명해주세요.{prompt_addition}{rag_prompt_addition}
    """

    try:
        if llm_provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            system_prompt = "당신은 숙련된 신약 개발 화학자입니다. 두 화합물의 구조-활성 관계(SAR)에 대한 분석을 요청받았습니다. 분석 결과를 전문가의 관점에서 명확하고 간결하게 마크다운 형식으로 작성해주세요."
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content, context_info
        
        elif llm_provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            full_prompt = "당신은 숙련된 신약 개발 화학자입니다. 다음 요청에 대해 전문가의 관점에서 명확하고 간결하게 마크다운 형식으로 답변해주세요.\n\n" + user_prompt
            response = model.generate_content(full_prompt)
            return response.text, context_info

    except Exception as e:
        st.error(f"{llm_provider} API 호출 중 오류 발생: {e}")
        return "가설 생성에 실패했습니다.", None
    
    return "알 수 없는 LLM 공급자입니다.", None
        

# --- QSAR 모델 로드 및 예측용 피처 생성 ---

@st.cache_data
def load_feature_list(path="sar-analysis-app/data/features.json"):
    """사전 훈련된 모델이 사용한 피처 목록을 불러옵니다."""
    try:
        with open(path, 'r') as f:
            features = json.load(f)
            return features, "피처 목록을 성공적으로 불러왔습니다."
    except FileNotFoundError:
        return None, f"오류: 피처 목록 파일({path})을 찾을 수 없습니다."
    except Exception as e:
        return None, f"피처 목록 로딩 중 오류 발생: {e}"
    
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

# --- SAR 분석용 함수들 ---
# def get_morgan_fingerprint(mol):
#     """(SAR 분석용) 분자 객체로부터 Morgan Fingerprint를 생성합니다."""
#     return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

# --- 기타 유틸리티 함수 ---

def smiles_to_descriptors(smiles, feature_list):
    """SMILES로부터 고정된 목록의 기술자 값을 계산합니다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # 기술자 이름과 함수를 매핑한 딕셔너리 생성
    descriptor_calculators = {name: func for name, func in Descriptors.descList}
    
    # feature_list 순서에 따라 값 계산
    descriptor_values = []
    for name in feature_list:
        try:
            # 해당 이름의 계산 함수를 찾아 실행
            value = descriptor_calculators[name](mol)
            descriptor_values.append(value)
        except:
            descriptor_values.append(0) # 오류 발생 시 0으로 처리

    return np.nan_to_num(np.array(descriptor_values), nan=0.0, posinf=0.0, neginf=0.0)

def find_activity_cliffs(df, similarity_threshold, activity_diff_threshold):
    """DataFrame에서 Activity Cliff 쌍을 찾고 스코어를 계산하여 정렬합니다."""
    # ... (기존 스코어링 로직과 동일)
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['fp'] = df['mol'].apply(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    df['scaffold'] = df['mol'].apply(lambda m: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
    
    cliffs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            sim = DataStructs.TanimotoSimilarity(df['fp'].iloc[i], df['fp'].iloc[j])
            if sim >= similarity_threshold:
                act_diff = abs(df['activity'].iloc[i] - df['activity'].iloc[j])
                if act_diff >= activity_diff_threshold:
                    score = act_diff * (sim - similarity_threshold) * (1 if df['scaffold'].iloc[i] == df['scaffold'].iloc[j] else 0.5)
                    cliffs.append({'mol_1': df.iloc[i], 'mol_2': df.iloc[j], 'similarity': sim, 'activity_diff': act_diff, 'score': score})
    cliffs.sort(key=lambda x: x['score'], reverse=True)
    return cliffs

def draw_molecule(smiles_string):
    encoded_smiles = quote(smiles_string)
    return f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/image?format=png&width=350&height=350"

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
