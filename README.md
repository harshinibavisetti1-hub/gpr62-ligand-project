# GPR62 Ligand Discovery Project

This project is a way to explore potential molecules related to **GPR62**, an orphan G protein-coupled receptor (GPCR).

The website combines published literature, related receptor information, and GPCRLigNet machine-learning predictions to help identify candidates that may be useful for future GPR62 research.

> The molecules shown in this project are potential candidates. A high machine-learning score does not confirm that a molecule binds to GPR62.

---

## Website Guide

### 1. Overview

The **Overview** page gives a quick summary of the project and shows the overall workflow.

The project followed this process:

**GPR62 Literature → Biological Entities → Related Receptors → Candidate Molecules → GPCRLigNet Prediction → Literature Evidence → Candidate Ranking**

---

### 2. GPR62 Literature

This page contains the articles collected about GPR62.

This was the starting point of the project

#### Main Columns

| Column | Meaning |
|---|---|
| **Year** | Year the article was published |
| **Title** | Name of the research article |
| **GPR62 Relevance** | How closely the article focuses on GPR62 |
| **Evidence Type** | Type of GPR62 information found in the article |
| **PMID** | PubMed identification number |
| **PMCID** | PubMed Central identification number, when available |
| **Journal** | Journal where the article was published |
| **DOI** | Digital identifier for the article |
| **Source** | Database where the article information was collected |

This page basically answers the question of ** what published research is available about GPR62?**

---

### 3. GPR62 Molecules & Proteins

This page shows important biological entities found in the GPR62 literature.

These can include:

- Proteins
- Genes
- Receptors
- Chemicals
- Hormones
- Signaling molecules
- Potential ligand candidates

#### Main Columns

| Column | Meaning |
|---|---|
| **Molecule / Protein** | Name of the biological |
| **Entity Type** | What type of entity it is |
| **Relationship / Role** | Why the entity is relevant to GPR62 |
| **Paper Count** | Number of papers connected to the entity |

For example, an entity may be classified as a protein, small molecule, second messenger, experimental compound, etc.

This page answers the question of **what molecules or type of molecules appear in the research?**

---

### 4. GPR62 vs. Related Receptors

Because GPR62 is an orphan receptor, there is limited information about what molecules directly interact with GPR62. This is why the project was expanded to look at molecules associated with other related receptors like GPR61, GPR135, MT2, and 5-HT6. These were determined by mentions in the GPR62 articles. These molecules aren't directly linked to GPR62, but could be looked into.

#### Main Columns

| Column | Meaning |
|---|---|
| **Molecule** | Candidate molecule |
| **Direct GPR62** | Whether direct GPR62 evidence was found |
| **Related Receptors** | Receptor(s) associated with the molecule |
| **GPCRLigNet Score** | Machine-learning prediction of GPCR-related activity |
| **Experimental Validation** | Strength of literature evidence for the receptor relationship |
| **Evidence Level** | Overall evidence category |


---

### 5. Ranked Candidates

This page shows the candidates that were prioritized using both machine-learning predictions and literature evidence.

#### Main Columns

| Column | Meaning |
|---|---|
| **Rank** | Position of the candidate in the final ranking |
| **Molecule** | Candidate molecule name |
| **Related Receptors** | Receptors connected to the candidate |
| **GPCRLigNet Score** | Predicted GPCR-related activity |
| **Experimental Validation** | Literature support for the receptor relationship |
| **Evidence Level** | Strength/category of the available evidence |
| **Priority** | Overall priority for further investigation |
| **PMIDs** | PubMed articles associated with the candidate |
| **PubChem CID** | PubChem identification number |
| **SMILES** | Text representation of the molecule's chemical structure |

A candidate with a high GPCRLigNet score is not necessarily a confirmed GPR62 ligand.

The ranking is used to identify molecules that may be worth looking into further.

---

## Evidence Levels

The page uses evidence levels to help separate stronger literature evidence from weaker or just computational evidence.

| Level | Simple Meaning |
|---|---|
| **A** | Direct experimental GPR62 evidence |
| **B** | Strong evidence involving a closely related receptor |
| **C** | Direct evidence involving another related receptor |
| **D** | Indirect or contextual evidence |
| **E** | Mainly computational or limited receptor-specific evidence |

The evidence level and GPCRLigNet score measure different things where the : 

**Evidence Level = literature support**

**GPCRLigNet Score = machine-learning prediction**

---

### 6. Predict New Molecule

This page allows to test a new molecule using GPCRLigNet.

Enter the molecule's **SMILES structure**, and the program converts the structure into a molecular fingerprint and sends it through the already trained GPCRLigNet model.

#### Main Results

| Result | Meaning |
|---|---|
| **GPCRLigNet Activity Score** | Predicted GPCR-related activity |
| **Inactive Probability** | Model prediction for inactivity |
| **Prediction Category** | Simple interpretation such as high, moderate, or low predicted activity |

A higher score means the model predicts stronger general GPCR-related activity. It does **not** mean the molecule has been proven to bind GPR62.

---

### 7. Batch Prediction

This page performs the same type of prediction for multiple molecules at once. For example, instead of entering one SMILES structure, multiple molecules can be analyzed together. This can be helpful for scanning a larger group of possible candidates together. 

| Column | Meaning |
|---|---|
| **Molecule** | Molecule name or identifier |
| **SMILES** | Chemical structure represented as text |
| **GPCRLigNet Score** | Machine-learning activity prediction |
| **Prediction** | Simple interpretation of the result |
| **Status** | Whether the molecule was successfully processed |

---

### 8. Evidence Database

This page has the larger dataset behind the candidate rankings.

Each row represents how many times the molecule appears more than once in the articles and how it's discussed in the research papers. 

| Column | Meaning |
|---|---|
| **Molecule** | Molecule or biological entity |
| **Related Receptor** | Receptor associated with the evidence |
| **GPCRLigNet Score** | Machine-learning prediction |
| **Experimental Validation** | Type/strength of literature evidence |
| **Evidence Level** | Evidence category |
| **Evidence Sentence** | Literature context supporting the relationship |
| **PMID** | PubMed article identifier |
| **PMCID** | PubMed Central identifier |
| **Year** | Publication year |
| **Article Title** | Name of the supporting article |
| **SMILES** | Chemical structure when available |
| **PubChem CID** | PubChem identifier when available |


---

### 9. About the Project

This page explains how the project was created and describes its limitations.

The project used:

- **PubMed / PubMed Central** for scientific literature
- **PubTator** for identifying biological entities
- **PubChem** for chemical information and SMILES structures
- **RDKit** for molecular fingerprints
- **GPCRLigNet** for machine-learning predictions
- **Streamlit** for the interactive website

---

## How to Interpret the Results
The project used literature evidence (looks at what previous research said about a molecule) and machine learning prediction (using GPCRLigNet to see how molecular structure relates to GPCR activity) to try and prioritize possible ligand candidates. The final candidates are **hypotheses for further investigation**, not experimentally confirmed GPR62 ligands.