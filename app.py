import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Therapeutic Peptide Miner", layout="wide")
st.title("In Silico Therapeutic Peptide Miner")
st.markdown("Isolate, profile, and perform preliminary computational screening on short peptides for predicted structural stability and target binding potential.")

# --- MODULE 1: INGESTION ---
def process_sequence(raw_data):
    """Failsafe parsing for DNA or Protein sequences."""
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
    """Cleavage and ExPASy/HemoPI algorithmic screening."""
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

    # ExPASy Instability
    df['Instability_Index'] = df['Sequence'].apply(lambda x: ProteinAnalysis(x).instability_index())
    df['Stability'] = np.where(df['Instability_Index'] <= 40.0, 'Stable', 'Unstable')

    # Protease Evasion (ExPASy PeptideCutter logic)
    def count_cuts(seq):
        return len(re.findall(r'[KR](?!P)', seq)) + len(re.findall(r'[FWYL](?!P)', seq)) + \
               len(re.findall(r'[AENDYFLIVW]', seq)) + len(re.findall(r'[LFVIAM]', seq))
    df['Cleavage_Sites'] = df['Sequence'].apply(count_cuts)
    
    # HemoPI SVM Simulation
    def calc_hemo(row):
        z = (row['Charge'] * 0.8) + (row['HydrophRatio'] * 1.5) - 2.5
        return round(1 / (1 + np.exp(-z)), 3)
    df['Hemo_PROB_Score'] = df.apply(calc_hemo, axis=1)
    
    conditions = [(df['Charge'] >= 2) & (df['HydrophRatio'] > 0.3), (df['Charge'] > 4) & (df['pI'] > 9.0)]
    df['Domain'] = np.select(conditions, ['AMP Potential', 'CPP Potential'], default='Therapeutic Candidate')
    
    # Isolate elite sequences
    elite_df = df[(df['Stability'] == 'Stable') & (df['Hemo_PROB_Score'] < 0.4) & (df['Cleavage_Sites'] < 10)]
    return elite_df.sort_values(by=['Hemo_PROB_Score', 'Instability_Index'])

# --- UI LAYOUT ---
with st.sidebar:
    st.header("Global Parameters")
    min_aa = st.slider("Minimum Peptide Length", 5, 10, 8)
    max_aa = st.slider("Maximum Peptide Length", 11, 20, 15)

st.subheader("1. Sequence Input Module")
input_method = st.radio("Select Data Source:", ("Text FASTA / Sequence", "Upload FASTA File"))

raw_fasta = ""
if input_method == "Text FASTA / Sequence":
    raw_fasta = st.text_area("Input Biological Sequence (DNA or Protein)", height=150)
else:
    uploaded_file = st.file_uploader("Upload .fasta format file", type=['fasta', 'txt'])
    if uploaded_file is not None:
        raw_fasta = uploaded_file.getvalue().decode("utf-8")

if 'elite_results' not in st.session_state:
    st.session_state.elite_results = pd.DataFrame()

if st.button("Execute Screening Protocol", type="primary") and raw_fasta:
    with st.spinner("Parsing data and auto-detecting sequence alphabet..."):
        proteins = process_sequence(raw_fasta)
    
    if not proteins:
        st.error("Error: No valid reading frames detected.")
    else:
        st.success(f"Successfully loaded sequence data.")
        with st.spinner("Applying in silico predictive thresholds (ExPASy/HemoPI)..."):
            st.session_state.elite_results = run_pipeline(proteins, min_aa, max_aa)

if not st.session_state.elite_results.empty:
    st.subheader("2. Elite Candidate Yield")
    st.info("The following sequences passed the *in silico* predictive thresholds for structural stability and simulated protease evasion.")
    
    display_cols = ['Sequence', 'Length', 'Charge', 'HydrophRatio', 'Instability_Index', 'Cleavage_Sites', 'Hemo_PROB_Score', 'Domain']
    st.dataframe(st.session_state.elite_results[display_cols])
    
    # Dynamic Formulation Warning for Cleavage Sites
    high_cleavage = st.session_state.elite_results[st.session_state.elite_results['Cleavage_Sites'] > 3]
    if not high_cleavage.empty:
        st.warning("**Formulation Note:** Sequences with >3 predicted cleavage sites will likely require chemical modification (e.g., D-amino acid substitution) or encapsulation in biocompatible nanocarriers (e.g., Chitosan nanoparticles, liposomes) to prevent rapid *in vivo* degradation.")
    
    st.markdown("---")
    
    # --- COMPREHENSIVE PROFILING (APD3 PARITY) ---
    st.subheader("3. Comprehensive Physicochemical Profiling")
    st.markdown("Select a candidate sequence to generate an isolated, APD3-style descriptor profile.")
    
    selected_pep = st.selectbox("Target Sequence for Profiling:", st.session_state.elite_results['Sequence'])
    
    if selected_pep:
        pa = ProteinAnalysis(selected_pep)
        selected_row = st.session_state.elite_results[st.session_state.elite_results['Sequence'] == selected_pep].iloc[0]
        
        # 1. Boman Index
        boman_dict = {
            'L': -4.92, 'I': -4.92, 'V': -3.04, 'F': -2.98, 'C': -2.87,
            'M': -2.35, 'A': -1.81, 'W': -0.92, 'G': 0.00, 'T': 1.08,
            'S': 1.13, 'Y': 1.15, 'P': 1.22, 'H': 2.33, 'N': 2.37,
            'Q': 2.37, 'D': 3.01, 'E': 3.14, 'K': 3.16, 'R': 3.19
        }
        boman_index = sum(boman_dict.get(aa, 0) for aa in selected_pep) / len(selected_pep)
        
        # 2. Extinction Coefficient
        w_count = selected_pep.count('W')
        y_count = selected_pep.count('Y')
        c_count = selected_pep.count('C')
        ext_coeff = (w_count * 5500) + (y_count * 1490) + ((c_count // 2) * 125)
        
        # 3. Rich Amino Acids
        counts = Counter(selected_pep)
        max_count = max(counts.values())
        rich_aas = ", ".join([aa for aa, count in counts.items() if count == max_count])

        # Dense Text Report
        st.markdown(f"**The total net charge** = {selected_row['Charge']:.2f}")
        st.markdown(f"**GRAVY** (Grand Average hydropathy value) = {pa.gravy():.3f}")
        st.markdown(f"**The molecular weight** of the input peptide = {pa.molecular_weight():.3f} Da")
        st.markdown(f"Assuming cysteines are paired, the **molar extinction coefficient** of the peptide = {ext_coeff}")
        st.markdown(f"**Protein-binding Potential (Boman index)** is: {boman_index:.2f} kcal/mol")
        st.markdown(f"**Your sequence is rich in:** {rich_aas}")
        st.markdown(f"**Instability index (II)** is computed to be {pa.instability_index():.2f}")

    st.markdown("---")
    
    # --- NOVELTY & DOWNSTREAM PROTOCOLS ---
    st.subheader("4. Novelty Verification & Structural Docking")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("**A. High-Throughput Novelty Screening**")
        st.info("Web-based BLAST queries throttle heavily for datasets >10 sequences. To verify true novelty for high-yield inputs, execute a Local BLAST+ query.")
        
        fasta_export = ""
        for index, row in st.session_state.elite_results.iterrows():
            fasta_export += f">Candidate_{index+1}|HemoScore_{row['Hemo_PROB_Score']}\n{row['Sequence']}\n"
            
        st.download_button(
            label="Download .FASTA for Local BLAST & 3D Modeling",
            data=fasta_export,
            file_name="elite_peptides.fasta",
            mime="text/plain"
        )
        
        st.markdown("""
        > **Terminal Protocol for Local Novelty Check:**
        > 1. Download the target database (e.g., APD3 or UniProt) locally.
        > 2. Format DB: `makeblastdb -in db.fasta -dbtype prot`
        > 3. Query: `blastp -query elite_peptides.fasta -db db.fasta -outfmt 6 -evalue 0.05`
        """)
        
    with col_v2:
        st.markdown("**B. In Silico Docking Readiness**")
        st.markdown(
            "Once sequence novelty is confirmed, utilize the exported `.fasta` file for structural and energetic profiling:\n\n"
            "*   **3D Coordinates:** Upload to **AlphaFold Server** or **RoseTTAFold** to predict high-accuracy tertiary structures (.pdb format).\n"
            "*   **Quantum Profiling:** Input `.pdb` geometries into **PySCF** to evaluate electrostatic potential surfaces and HOMO/LUMO energy gaps.\n"
            "*   **Molecular Docking:** Convert structures to `.pdbqt` format for binding affinity calculations against therapeutic targets using **AutoDock Vina**."
        )
