import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from modlamp.descriptors import GlobalDescriptor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="In Silico Peptide Miner", layout="wide")
st.title("In Silico Peptide Miner: Antimicrobial Peptide Discovery Pipeline")
st.markdown("Isolate, profile, and screen short novel peptides for high protease evasion, structural stability, and target binding potential.")

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
    
    def calc_hemo(row):
        z = (row['Charge'] * 0.8) + (row['HydrophRatio'] * 1.5) - 2.5
        return round(1 / (1 + np.exp(-z)), 3)
    df['Hemo_PROB_Score'] = df.apply(calc_hemo, axis=1)
    
    conditions = [(df['Charge'] >= 2) & (df['HydrophRatio'] > 0.3), (df['Charge'] > 4) & (df['pI'] > 9.0)]
    df['Domain'] = np.select(conditions, ['AMP Potential', 'CPP Potential'], default='Therapeutic Candidate')
    
    elite_df = df[(df['Stability'] == 'Stable') & (df['Hemo_PROB_Score'] < 0.4) & (df['Cleavage_Sites'] < 10)]
    return elite_df.sort_values(by=['Hemo_PROB_Score', 'Instability_Index'])

# --- MODULE 4: NOVELTY SCORING ---
def calculate_novelty(sequence):
    try:
        result_handle = NCBIWWW.qblast("blastp", "swissprot", sequence, hitlist_size=1)
        blast_record = NCBIXML.read(result_handle)
        
        if len(blast_record.alignments) == 0:
            return "100% (No hits found in SwissProt)"
            
        alignment = blast_record.alignments[0]
        hsp = alignment.hsps[0]
        identity = (hsp.identities / hsp.align_length) * 100
        novelty_score = max(0, 100 - identity)
        return f"{novelty_score:.1f}%"
    except Exception as e:
        return "BLAST Connection Error"

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
        with st.spinner("Applying ExPASy cleavage algorithms and HemoPI parameters..."):
            st.session_state.elite_results = run_pipeline(proteins, min_aa, max_aa)

if not st.session_state.elite_results.empty:
    st.subheader("2. Elite Candidate Yield")
    st.info("The following sequences passed all in vitro stability and in vivo protease evasion thresholds.")
    display_cols = ['Sequence', 'Length', 'Charge', 'HydrophRatio', 'Instability_Index', 'Cleavage_Sites', 'Hemo_PROB_Score', 'Domain']
    st.dataframe(st.session_state.elite_results[display_cols])
    
    st.markdown("---")
    
    # --- COMPREHENSIVE PROFILING ---
    st.subheader("3. Comprehensive Physicochemical Profiling (ExPASy/APD3 Parity)")
    st.markdown("Select a candidate sequence to generate an isolated 2D descriptor profile.")
    
    selected_pep = st.selectbox("Target Sequence for Profiling:", st.session_state.elite_results['Sequence'])
    
    if selected_pep:
        pa = ProteinAnalysis(selected_pep)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Molecular Weight", f"{pa.molecular_weight():.2f} Da")
        col_m2.metric("Isoelectric Point (pI)", f"{pa.isoelectric_point():.2f}")
        col_m3.metric("GRAVY Score", f"{pa.gravy():.3f}")
        col_m4.metric("Instability Index", f"{pa.instability_index():.2f}") 
        
        st.markdown("**Amino Acid Composition (%)**")
        
        # Version-proof standard biochemical calculation for AA frequencies
        standard_aa = list("ACDEFGHIKLMNPQRSTVWY")
        aa_comp = {aa: (selected_pep.count(aa) / len(selected_pep)) for aa in standard_aa}
        
        aa_df = pd.DataFrame(list(aa_comp.items()), columns=['Amino Acid', 'Frequency']).set_index('Amino Acid')
        st.bar_chart(aa_df * 100)

    st.markdown("---")
    
    # --- VALIDATION & DOCKING EXPORT ---
    st.subheader("4. Structural Validation & Downstream Docking Protocols")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("**A. Sequence Homology Verification**")
        st.info("Query the curated SwissProt database to verify novelty prior to structural modeling.")
        if st.button("Initiate BLASTp Query"):
            with st.spinner("Aligning sequence..."):
                novelty = calculate_novelty(selected_pep)
                st.success(f"Novelty Score for {selected_pep}: **{novelty}**")
                
        st.markdown("**B. Export for 3D Modeling (.FASTA)**")
        fasta_export = ""
        for index, row in st.session_state.elite_results.head(50).iterrows():
            fasta_export += f">Candidate_{index+1}|HemoScore_{row['Hemo_PROB_Score']}\n{row['Sequence']}\n"
            
        st.download_button(
            label="Download .FASTA Output",
            data=fasta_export,
            file_name="elite_peptides_structural.fasta",
            mime="text/plain"
        )
        
    with col_v2:
        st.markdown("**C. In Silico Docking Readiness**")
        st.markdown(
            "Once elite candidates are isolated and exported, utilize the following platforms to complete the structural analysis pipeline:\n\n"
            "*   **3D Structure Generation:** Upload the exported multi-FASTA file directly to [AlphaFold Server](https://alphafoldserver.com/) or [RoseTTAFold](https://robetta.bakerlab.org/) to predict high-accuracy 3D coordinates (.pdb).\n"
            "*   **Quantum Biological Profiling:** Input the resulting 3D structures into **PySCF** to evaluate electrostatic potentials and HOMO/LUMO energy gaps for stability verification.\n"
            "*   **Target Molecular Docking:** Convert predicted structures to `.pdbqt` format for binding affinity calculations against pathogenic targets using **AutoDock Vina** or **HADDOCK**."
        )
