# 🧬 In Silico Peptide Miner: Advanced Antimicrobial Peptide Discovery Pipeline

An automated, high-throughput bioinformatics pipeline designed to isolate, profile, and screen short, novel peptides from raw biological sequences. This tool bridges the gap between raw FASTA data and structural biology by employing gold-standard physicochemical screening algorithms.

## 🔬 Academic Scope & Biochemical Logic
This pipeline was developed to streamline the screening phase of peptide discovery. It translates physiological survival parameters into strict computational filters, ensuring that only highly viable candidates proceed to 3D modeling and molecular docking. 

The screening mechanism integrates the following established models:
*   **Economical Cleavage:** Utilizes a sliding window approach to isolate 8–15 amino acid fragments, optimizing for cost-effective lab synthesis and membrane permeability.
*   **Test-Tube Stability (ExPASy ProtParam):** Calculates the Guruprasad Instability Index to ensure in vitro survival (Index < 40).
*   **Protease Evasion (ExPASy PeptideCutter Logic):** Scans sequences for cleavage motifs targeted by major proteolytic enzymes (Trypsin, Chymotrypsin, Proteinase K, and Thermolysin).
*   **Hemotoxicity Scoring (HemoPI SVM Simulation):** Evaluates the lethal combination of net positive charge and extreme hydrophobicity (which ruptures red blood cells) using a simulated logistic function:
    $$ P = \frac{1}{1 + e^{-z}} $$
    *(Where $z$ is the weighted sum of charge and hydrophobic ratio).*
*   **Sequence Novelty (NCBI BLAST):** Queries the non-redundant database to verify sequence novelty, ensuring isolated fragments represent unpatented, uncharacterized intellectual property.

## 🚀 Usage & Deployment

### Local Execution
To run this pipeline on your local machine, ensure you have Python installed, clone this repository, and install the required dependencies:
```bash
pip install -r requirements.txt
streamlit run app.py
