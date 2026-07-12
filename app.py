import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import time
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from modlamp.descriptors import GlobalDescriptor
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="In Silico Peptide Miner", layout="wide")
st.title("🧬 Advanced Antimicrobial Peptide Mining Pipeline")
st.markdown("Isolate, profile, and screen short novel peptides for high protease evasion and structural stability.")

# --- MODULE 1: INGESTION ---
def process_sequence(raw_data):
    seq_str = ""
    if ">" in raw_data:
        records = list(SeqIO.parse(io.StringIO(raw_data.strip()), "fasta"))
        if records and str(records[0].seq).strip():
            seq_str = str(records[0].seq).strip().upper()
    if not seq_str:
        clean_text = re.sub(r'[^a-zA-Z]', ' ', raw_data)
        words = clean_text.split()
        if words:
            seq_str = max(words, key=len).upper()
            
    if not seq_str: return []
    
    is_protein = not set(seq_str).issubset(set("ACGTNU"))
    proteins = [seq_str] if is_protein else []
    
    if not is_protein:
        seq_obj = Seq(seq_str)
        for strand, nuc in [(1, seq_obj), (-1, seq_obj.reverse_complement())]:
            for frame in range(3):
                trim_len = len(nuc[frame:]) - (len(nuc[frame:]) % 3)
                translated = nuc[frame:frame+trim_len].translate(to_stop=False)
                for prot in str(translated).split('*'):
                    if len(prot) >= 30:
                        proteins.append(prot)
    return proteins

# --- MODULE 2 & 3: PROFILING & SCORING ---
@st.cache_data
def run_pipeline(protein_list, min_len, max_len):
    peptides = set()
    for seq in protein_list:
        for length in range(min_len, max_len + 1):
            for i in range(len(seq) - length + 1):
                peptides.add(seq[i:i+length])
    
    pep_list = list(peptides)
    if not pep_list: return pd.DataFrame()
    
    desc = GlobalDescriptor(pep_list)
    desc.calculate_all()
    df = pd.DataFrame(desc.descriptor, columns=desc.featurenames)
    df.insert(0, 'Sequence', pep_list)
    df['Length'] = df['Sequence'].apply(len)
    
    df = df[(df['Charge'] > 0) & (df['HydrophRatio'] > -0.5)].copy()
    if df.empty: return df

    df['Instability_Index'] = df['Sequence'].apply(lambda x: ProteinAnalysis(x).instability_index())
    df['Stability'] = np.where(df['Instability_Index'] <= 40.0, 'Stable', 'Unstable')

    def count_cuts(seq):
        return len(re.findall(r'[KR](?!P)', seq)) + len(re.findall(r'[FWYL](?!P)', seq)) + \
               len(re.findall(r'[AENDYFLIVW]', seq)) + len(re.findall(r'[LFVIAM]', seq))
    df['Cleavage_Sites'] = df['Sequence'].apply(count_cuts)
    
    # ML Integration Architecture (Ready for DBAASP/ENA trained models)
    # try:
    #     model = joblib.load('models/hemo_rf_model.pkl')
    #     features = df[['Charge', 'HydrophRatio', 'pI']].values
    #     df['Hemo_PROB_Score'] = model.predict_proba(features)[:, 1]
    # except FileNotFoundError:
    # Simulated logistic function fallback
    def calc_hemo(row):
        z = (row['Charge'] * 0.8) + (row['HydrophRatio'] * 1.5) - 2.5
        return round(1 / (1 + np.exp(-z)), 3)
    df['Hemo_PROB_Score'] = df.apply(calc_hemo, axis=1)
    
    conditions = [(df['Charge'] >= 2) & (df['HydrophRatio'] > 0.3), (df['Charge'] > 4) & (df['pI'] > 9.0)]
    df['Domain'] = np.select(conditions, ['AMP Potential', 'CPP Potential'], default='Therapeutic')
    
    elite_df = df[(df['Stability'] == 'Stable') & (df['Hemo_PROB_Score'] < 0.4) & (df['Cleavage_Sites'] < 10)]
    return elite_df.sort_values(by=['Hemo_PROB_Score', 'Instability_Index'])

# --- MODULE 4: NOVELTY SCORING ---
def calculate_novelty(sequence):
    """Queries NCBI to find sequence homology. High E-value = High Novelty."""
    try:
        result_handle = NCBIWWW.qblast("blastp", "nr", sequence, hitlist_size=1)
        blast_record = NCBIXML.read(result_handle)
        
        if len(blast_record.alignments) == 0:
            return "100% (No hits found)"
            
        alignment = blast_record.alignments[0]
        hsp = alignment.hsps[0]
        identity = (hsp.identities / hsp.align_length) * 100
        
        # If it's highly identical to a known sequence, novelty is low
        novelty_score = max(0, 100 - identity)
        return f"{novelty_score:.1f}%"
    except Exception as e:
        return "BLAST Error"

# --- UI LAYOUT ---
with st.sidebar:
    st.header("⚙️ Pipeline Parameters")
    min_aa = st.slider("Min Peptide Length", 5, 10, 8)
    max_aa = st.slider("Max Peptide Length", 11, 20, 15)

st.subheader("1. Sequence Input")
input_method = st.radio("Choose Input Method:", ("Paste FASTA/Sequence", "Upload FASTA File"))

raw_fasta = ""
if input_method == "Paste FASTA/Sequence":
    raw_fasta = st.text_area("Input Data", height=150)
else:
    uploaded_file = st.file_uploader("Upload .fasta file", type=['fasta', 'txt'])
    if uploaded_file is not None:
        raw_fasta = uploaded_file.getvalue().decode("utf-8")

if 'elite_results' not in st.session_state:
    st.session_state.elite_results = pd.DataFrame()

if st.button("Run Screening Pipeline", type="primary") and raw_fasta:
    with st.spinner("Parsing sequence and auto-detecting format..."):
        proteins = process_sequence(raw_fasta)
    
    if not proteins:
        st.error("No valid sequence detected.")
    else:
        st.success(f"Detected {len(proteins)} valid reading frame(s).")
        with st.spinner("Cleaving, profiling, and simulating ML algorithms..."):
            st.session_state.elite_results = run_pipeline(proteins, min_aa, max_aa)

if not st.session_state.elite_results.empty:
    st.subheader("2. Elite Candidate Results")
    st.success(f"Isolated {len(st.session_state.elite_results)} highly evasive candidates.")
    display_cols = ['Sequence', 'Length', 'Charge', 'HydrophRatio', 'Instability_Index', 'Cleavage_Sites', 'Hemo_PROB_Score', 'Domain']
    st.dataframe(st.session_state.elite_results[display_cols])
    
    st.subheader("3. Validation & Structural Export")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sequence Novelty Check**")
        st.info("BLAST queries take 15-30 seconds per sequence. Select a top candidate to check.")
        target_seq = st.selectbox("Select Peptide for Novelty Check:", st.session_state.elite_results['Sequence'].head(5))
        if st.button("Query NCBI BLAST"):
            with st.spinner("Aligning against non-redundant database..."):
                novelty = calculate_novelty(target_seq)
                st.success(f"Novelty Score for {target_seq}: **{novelty}**")
                
    with col2:
        st.markdown("**Export for 3D Modeling (AlphaFold)**")
        # Generate multi-FASTA format for structural bridge
        fasta_export = ""
        for index, row in st.session_state.elite_results.head(10).iterrows():
            fasta_export += f">Elite_Candidate_{index+1}|Score_{row['Hemo_PROB_Score']}\n{row['Sequence']}\n"
            
        st.download_button(
            label="Download .FASTA for AlphaFold",
            data=fasta_export,
            file_name="elite_peptides_for_3D_modeling.fasta",
            mime="text/plain"
        )
