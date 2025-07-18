import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
import google.generativeai as genai
from urllib.parse import quote
import joblib

# --- Phase 1: 데이터 준비 및 탐색 ---

@st.cache_data
def load_data(uploaded_file):
    """업로드된 파일을 Pandas DataFrame으로 로드합니다."""
    if uploaded_file is not None:
        try:
            # 파일 경로에 따라 적절한 로더 사용
            if isinstance(uploaded_file, str) and ("sample_data.csv" in uploaded_file or "large_sar_data.csv" in uploaded_file):
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

# --- QSAR 모델 로드 및 예측용 피처 생성 ---
FIXED_DESCRIPTOR_NAMES = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt',
    'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge',
    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1',
    'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3',
    'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3',
    'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3',
    'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
    'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1',
    'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
    'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
    'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
    'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
    'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_no_H',
    'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_no_H', 'fr_C_S', 'fr_HOCCN',
    'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
    'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide',
    'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
    'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
    'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
    'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
    'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
    'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
    'fr_unbrch_alkane', 'fr_urea'
]

def smiles_to_descriptors(smiles):
    """(QSAR 예측용) SMILES 문자열로부터 고정된 목록의 RDKit 기술자를 계산합니다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptor_values = []
    for desc_name in FIXED_DESCRIPTOR_NAMES:
        try:
            func = getattr(Descriptors, desc_name)
            descriptor_values.append(func(mol))
        except AttributeError:
            descriptor_values.append(np.nan)
            
    return np.nan_to_num(np.array(descriptor_values), nan=0.0, posinf=0.0, neginf=0.0)

@st.cache_resource
def load_pretrained_model(model_path="qsar_model.joblib"):
    """사전 훈련된 QSAR 모델 파일을 불러옵니다."""
    try:
        model_pipeline = joblib.load(model_path)
        return model_pipeline, "사전 훈련된 QSAR 모델을 성공적으로 불러왔습니다."
    except FileNotFoundError:
        return None, f"오류: '{model_path}' 파일을 찾을 수 없습니다."
    except Exception as e:
        return None, f"모델 로딩 중 오류 발생: {e}"

# --- SAR 분석용 함수들 ---

def get_morgan_fingerprint(mol):
    """(SAR 분석용) 분자 객체로부터 Morgan Fingerprint를 생성합니다."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

@st.cache_data
def prepare_comparison_data(_df):
    df = _df.copy()
    df.dropna(subset=['SMILES'], inplace=True)
    df['SMILES'] = df['SMILES'].astype(str)
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df.dropna(subset=['mol'], inplace=True)
    df['fp'] = df['mol'].apply(get_morgan_fingerprint)
    return df

def find_most_similar_compounds(new_smiles, training_df, top_n=2):
    new_mol = Chem.MolFromSmiles(new_smiles)
    if not new_mol: return []
    new_fp = get_morgan_fingerprint(new_mol)
    training_df['similarity'] = training_df['fp'].apply(lambda x: DataStructs.TanimotoSimilarity(x, new_fp))
    most_similar = training_df.sort_values(by='similarity', ascending=False).head(top_n)
    return most_similar.to_dict('records')

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
    prompt = f"당신은 숙련된 신약 개발 화학자입니다. 두 화합물의 구조-활성 관계(SAR)에 대한 분석을 요청받았습니다.\n\n**분석 대상:**\n- **화합물 A (낮은 활성):**\n  - ID: {compound_a['ID']}\n  - SMILES: {compound_a['SMILES']}\n  - 활성도 (pKi): {compound_a['activity']:.2f}\n- **화합물 B (높은 활성):**\n  - ID: {compound_b['ID']}\n  - SMILES: {compound_b['SMILES']}\n  - 활성도 (pKi): {compound_b['activity']:.2f}\n\n**분석 요청:**\n두 화합물은 구조적으로 매우 유사하지만(Tanimoto 유사도: {cliff['similarity']:.2f}), 활성도에서 큰 차이(pKi 차이: {cliff['activity_diff']:.2f})를 보입니다.\n이러한 'Activity Cliff' 현상을 유발하는 핵심적인 구조적 차이점을 찾아내고, 그 차이가 어떻게 활성도 증가로 이어졌는지에 대한 화학적 가설을 전문가의 관점에서 설명해주세요."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API 호출 중 오류 발생: {e}")
        return "가설 생성에 실패했습니다."

def draw_molecule(smiles_string):
    encoded_smiles = quote(smiles_string)
    return f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/image?format=png&width=350&height=350"
