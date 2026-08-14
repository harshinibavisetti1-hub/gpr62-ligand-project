# GPR62 Ligand Discovery Platform

An interactive literature-mining and machine-learning workflow for prioritizing potential molecules relevant to the orphan G protein-coupled receptor GPR62.

## Features

- Direct GPR62 literature analysis
- PubMed and PMC literature integration
- Biomedical entity extraction
- GPR62-associated molecule and protein analysis
- Related receptor analysis:
  - GPR61
  - GPR135
  - MT2 receptor
  - 5-HT6 receptor
- GPCRLigNet machine-learning predictions
- Literature evidence classification
- Candidate prioritization
- Interactive single-molecule prediction
- Batch SMILES prediction
- Full molecule-paper evidence database
- Downloadable results

## Machine-Learning Workflow

SMILES  
→ RDKit molecular structure  
→ Morgan fingerprint (radius 4, 1024 bits)  
→ GPCRLigNet  
→ predicted GPCR activity score

## Important Interpretation

The GPCRLigNet activity score predicts general GPCR-related activity.

A high GPCRLigNet score does **not** establish direct binding to GPR62 and should not be interpreted as a probability of GPR62 binding.

Related-receptor literature and machine-learning predictions are used to prioritize molecules for future experimental validation.

## Main Data Sections

- Direct GPR62 literature
- Direct GPR62 molecules/proteins
- GPR62 vs related receptors
- Ranked candidate molecules
- Paper-level evidence database

## Running Locally

Activate the Python environment and run:

```bash
streamlit run streamlit_app.py