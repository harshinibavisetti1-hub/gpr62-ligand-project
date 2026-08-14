import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow import keras
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GPR62 Ligand Discovery",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# FILE LOADING
# ============================================================

@st.cache_data
def load_csv(path):

    if os.path.exists(path):

        try:
            return pd.read_csv(path)

        except Exception as e:
            st.error(
                f"Could not load {path}: {e}"
            )
            return pd.DataFrame()

    return pd.DataFrame()


direct_articles = load_csv(
    "gpr62_direct_articles_classified.csv"
)

direct_entities = load_csv(
    "gpr62_direct_entity_summary.csv"
)

comparison = load_csv(
    "gpr62_vs_related_receptors_final.csv"
)

top_candidates = load_csv(
    "gpr62_top_candidates.csv"
)

molecule_summary = load_csv(
    "gpr62_final_molecule_summary.csv"
)

master = load_csv(
    "gpr62_master_dataset_enriched.csv"
)


# ============================================================
# MOLECULE HELPERS
# ============================================================

def valid_value(value):

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except:
        pass

    return bool(
        str(value).strip()
    )


def safe_text(value, default="N/A"):

    if not valid_value(value):
        return default

    return str(value)


def safe_float(value):

    try:
        if pd.isna(value):
            return None

        return float(value)

    except:
        return None


def smiles_to_image(smiles):

    if not valid_value(smiles):
        return None

    try:

        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        if mol is None:
            return None

        return Draw.MolToImage(
            mol,
            size=(450, 330)
        )

    except Exception:
        return None


def canonical_smiles(smiles):

    if not valid_value(smiles):
        return None

    try:

        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:
        return None


# ============================================================
# GPCRLIGNET
# ============================================================

def crossentropy(
    y_true,
    y_pred
):

    return tf.reduce_mean(

        -1.0
        * y_true[:, 0]
        * tf.math.log(

            tf.clip_by_value(
                y_pred[:, 0],
                1e-10,
                1.0
            )
        )

        -

        y_true[:, 1]
        * tf.math.log(

            tf.clip_by_value(
                y_pred[:, 1],
                1e-10,
                1.0
            )
        )
    )


@st.cache_resource
def load_gpcrlignet():

    model_path = (
        "models/"
        "cicular_4_models_6_17_21/"
        "model_cicular4.tf"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            "GPCRLigNet model folder was not found: "
            + model_path
        )

    model = keras.models.load_model(
        model_path,
        custom_objects={
            "crossentropy":
                crossentropy
        }
    )

    return model


def generate_gpcr_fingerprint(
    smiles
):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None


    # Match original GPCRLigNet
    # preprocessing exactly.

    mol_with_h = Chem.AddHs(
        mol
    )


    fingerprint = (
        AllChem
        .GetMorganFingerprintAsBitVect(

            mol_with_h,

            4,

            nBits=1024
        )
    )


    arr = np.zeros(
        (1024,),
        dtype=np.float32
    )


    DataStructs.ConvertToNumpyArray(
        fingerprint,
        arr
    )


    return arr


def predict_gpcr_activity(
    smiles
):

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:

        return {
            "success": False,
            "error":
                "Invalid SMILES string."
        }


    fingerprint = (
        generate_gpcr_fingerprint(
            smiles
        )
    )


    if fingerprint is None:

        return {
            "success": False,
            "error":
                "Could not generate molecular fingerprint."
        }


    input_data = (
        fingerprint.reshape(
            1,
            1024
        )
    )


    try:

        model = (
            load_gpcrlignet()
        )


        prediction = (
            model.predict(
                input_data,
                verbose=0
            )
        )


        activity_score = float(
            prediction[0][0]
        )


        inactive_score = float(
            prediction[0][1]
        )


    except Exception as e:

        return {
            "success": False,
            "error":
                f"Model prediction failed: {e}"
        }


    return {

        "success":
            True,

        "activity_score":
            activity_score,

        "inactive_score":
            inactive_score,

        "raw_prediction":
            prediction.tolist()
    }


def activity_label(score):

    if score >= 0.80:
        return (
            "High predicted GPCR activity"
        )

    elif score >= 0.50:
        return (
            "Moderate predicted GPCR activity"
        )

    else:
        return (
            "Low predicted GPCR activity"
        )


# ============================================================
# DATABASE LOOKUP
# ============================================================

def find_existing_molecule(
    smiles
):

    if molecule_summary.empty:
        return None

    if (
        "SMILES"
        not in molecule_summary.columns
    ):
        return None


    query = canonical_smiles(
        smiles
    )


    if query is None:
        return None


    for _, row in (
        molecule_summary.iterrows()
    ):

        stored_smiles = row.get(
            "SMILES"
        )


        if not valid_value(
            stored_smiles
        ):
            continue


        stored = canonical_smiles(
            stored_smiles
        )


        if stored == query:
            return row


    return None


def find_master_evidence(
    molecule_name
):

    if master.empty:
        return pd.DataFrame()

    if (
        "Molecule"
        not in master.columns
    ):
        return pd.DataFrame()


    return master[

        master["Molecule"]
        .astype(str)
        .str.lower()

        ==

        str(
            molecule_name
        ).lower()

    ].copy()


# ============================================================
# PMID LINKS
# ============================================================

def show_pmid_links(
    pmids
):

    if not valid_value(pmids):
        return


    values = [

        x.strip()

        for x in
        str(pmids)
        .replace(",", ";")
        .split(";")

        if x.strip()
    ]


    if not values:
        return


    links = [

        f"[PMID {p}]"
        f"(https://pubmed.ncbi.nlm.nih.gov/{p}/)"

        for p in values
    ]


    st.markdown(
        " | ".join(
            links
        )
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧬 GPR62 Ligand Discovery Platform"
)

st.caption(
    "Literature mining + related-receptor analysis + "
    "GPCRLigNet candidate prioritization"
)

st.warning(
    "GPCRLigNet predicts general GPCR-related activity. "
    "A high score does NOT mean that a molecule has a "
    "high probability of binding specifically to GPR62."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "GPR62 Explorer"
)


page = st.sidebar.radio(

    "View",

    [

        "Overview",

        "Direct GPR62 Literature",

        "Direct Molecules / Proteins",

        "GPR62 vs Other Receptors",

        "Ranked Candidates",

        "Predict New Molecule",

        "Batch Prediction",

        "Full Evidence Database"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header(
        "Project Overview"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Direct GPR62 articles",
        len(direct_articles)
    )


    col2.metric(
        "Direct entities",
        len(direct_entities)
    )


    col3.metric(
        "Comparison entities",
        len(comparison)
    )


    col4.metric(
        "Ranked candidates",
        len(top_candidates)
    )


    st.subheader(
        "Discovery Workflow"
    )


    st.code(
        """
PubMed + PMC
      ↓
Literature collection
      ↓
PubTator biomedical entity extraction
      ↓
Direct GPR62 evidence analysis
      ↓
Related receptor expansion
(GPR61, GPR135, MT2, 5-HT6)
      ↓
Chemical extraction
      ↓
PubChem structure / SMILES retrieval
      ↓
Morgan fingerprints
(radius 4, 1024 bits)
      ↓
GPCRLigNet prediction
      ↓
Literature evidence validation
      ↓
Candidate ranking
        """,
        language="text"
    )


    st.subheader(
        "Why Related Receptors?"
    )


    st.write(
        "Because GPR62 is an orphan receptor with limited "
        "direct ligand information, related receptors were "
        "used to expand the candidate search space. "
        "Candidate molecules from these receptors were then "
        "evaluated computationally with GPCRLigNet."
    )


    st.subheader(
        "Evidence Levels"
    )


    evidence_table = pd.DataFrame({

        "Level":
            ["A", "B", "C", "D", "E"],

        "Meaning": [

            (
                "Direct experimental "
                "GPR62 evidence"
            ),

            (
                "Strong evidence involving "
                "a closely related receptor"
            ),

            (
                "Direct receptor evidence "
                "at another related GPCR"
            ),

            (
                "Indirect or contextual "
                "receptor evidence"
            ),

            (
                "ML prediction with limited "
                "receptor-specific validation"
            )
        ]
    })


    st.dataframe(
        evidence_table,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "The evidence categories and priority ranking are "
        "project-specific interpretation layers and are "
        "separate from the raw GPCRLigNet prediction score."
    )


# ============================================================
# DIRECT GPR62 LITERATURE
# ============================================================

elif page == (
    "Direct GPR62 Literature"
):

    st.header(
        "Direct GPR62 Literature"
    )


    st.write(
        "These records represent the original direct "
        "GPR62 PubMed literature search."
    )


    if direct_articles.empty:

        st.error(
            "Direct GPR62 article dataset was not found."
        )

    else:

        filtered = (
            direct_articles.copy()
        )


        if (
            "Article Status"
            in filtered.columns
        ):

            statuses = sorted(

                filtered[
                    "Article Status"
                ]
                .dropna()
                .unique()
            )


            selected_status = (
                st.multiselect(

                    "Article status",

                    statuses,

                    default=statuses
                )
            )


            filtered = filtered[

                filtered[
                    "Article Status"
                ].isin(
                    selected_status
                )
            ]


        search = st.text_input(
            "Search direct GPR62 literature"
        )


        if search:

            mask = (

                filtered
                .astype(str)
                .apply(

                    lambda row:

                    row.str.contains(

                        search,

                        case=False,

                        na=False

                    ).any(),

                    axis=1
                )
            )


            filtered = (
                filtered[mask]
            )


        display_columns = [

            c for c in [

                "Year",

                "Title",

                "Article Status",

                "GPR62 Relevance",

                "Evidence Type",

                "Important Terms",

                "PMID",

                "PMCID",

                "Source",

                "Journal",

                "Authors",

                "DOI",

                "PubMed Link",

                "PMC Link"

            ]

            if c in
            filtered.columns
        ]


        st.write(
            "Articles:",
            len(filtered)
        )


        column_config = {}


        if (
            "PubMed Link"
            in display_columns
        ):

            column_config[
                "PubMed Link"
            ] = (
                st.column_config
                .LinkColumn(
                    "PubMed"
                )
            )


        if (
            "PMC Link"
            in display_columns
        ):

            column_config[
                "PMC Link"
            ] = (
                st.column_config
                .LinkColumn(
                    "PMC"
                )
            )


        st.dataframe(

            filtered[
                display_columns
            ],

            use_container_width=True,

            hide_index=True,

            column_config=
                column_config
        )


        st.download_button(

            "Download direct GPR62 articles",

            filtered.to_csv(
                index=False
            ),

            "gpr62_direct_articles_filtered.csv",

            "text/csv"
        )


# ============================================================
# DIRECT MOLECULES / PROTEINS
# ============================================================

elif page == (
    "Direct Molecules / Proteins"
):

    st.header(
        "Direct GPR62 Molecules / Proteins"
    )


    st.write(
        "These entities were identified from the direct "
        "GPR62 literature and curated by their relationship "
        "to GPR62."
    )


    if direct_entities.empty:

        st.error(
            "Direct entity dataset was not found."
        )

    else:

        filtered = (
            direct_entities.copy()
        )


        if (
            "Entity Class"
            in filtered.columns
        ):

            classes = sorted(

                filtered[
                    "Entity Class"
                ]
                .dropna()
                .unique()
            )


            selected = (
                st.multiselect(

                    "Entity class",

                    classes,

                    default=classes
                )
            )


            filtered = filtered[

                filtered[
                    "Entity Class"
                ].isin(selected)
            ]


        st.dataframe(

            filtered,

            use_container_width=True,

            hide_index=True
        )


        st.download_button(

            "Download direct GPR62 entities",

            filtered.to_csv(
                index=False
            ),

            "gpr62_direct_entities_filtered.csv",

            "text/csv"
        )


# ============================================================
# GPR62 VS OTHER RECEPTORS
# ============================================================

elif page == (
    "GPR62 vs Other Receptors"
):

    st.header(
        "GPR62 Molecules vs Other Receptors"
    )


    st.write(
        "Compare direct GPR62 evidence with molecules "
        "identified from GPR61, GPR135, MT2, and 5-HT6 "
        "literature."
    )


    if comparison.empty:

        st.error(
            "Comparison dataset was not found."
        )

    else:

        filtered = (
            comparison.copy()
        )


        search = st.text_input(
            "Search molecule, protein, or receptor"
        )


        if search:

            mask = (

                filtered
                .astype(str)
                .apply(

                    lambda row:

                    row.str.contains(

                        search,

                        case=False,

                        na=False

                    ).any(),

                    axis=1
                )
            )


            filtered = (
                filtered[mask]
            )


        col1, col2 = (
            st.columns(2)
        )


        with col1:

            direct_only = (
                st.checkbox(
                    "Direct GPR62 evidence only"
                )
            )


        with col2:

            scored_only = (
                st.checkbox(
                    "Only molecules with GPCRLigNet scores"
                )
            )


        if (
            direct_only
            and
            "Direct GPR62 Evidence"
            in filtered.columns
        ):

            filtered = filtered[

                filtered[
                    "Direct GPR62 Evidence"
                ]
                .astype(str)
                .str.lower()

                == "yes"
            ]


        if (
            scored_only
            and
            "GPCRLigNet Score"
            in filtered.columns
        ):

            filtered[
                "GPCRLigNet Score"
            ] = pd.to_numeric(

                filtered[
                    "GPCRLigNet Score"
                ],

                errors="coerce"
            )


            filtered = filtered[

                filtered[
                    "GPCRLigNet Score"
                ].notna()
            ]


        st.write(
            "Results:",
            len(filtered)
        )


        st.dataframe(

            filtered,

            use_container_width=True,

            hide_index=True
        )


        st.download_button(

            "Download receptor comparison",

            filtered.to_csv(
                index=False
            ),

            "gpr62_vs_related_filtered.csv",

            "text/csv"
        )


# ============================================================
# RANKED CANDIDATES
# ============================================================

elif page == (
    "Ranked Candidates"
):

    st.header(
        "Ranked GPR62 Candidate Molecules"
    )


    st.write(
        "Candidate ranking combines GPCRLigNet predictions "
        "with literature and receptor-relevance evidence."
    )


    if top_candidates.empty:

        st.error(
            "Candidate ranking dataset was not found."
        )

    else:

        filtered = (
            top_candidates.copy()
        )


        if (
            "Evidence Level"
            in filtered.columns
        ):

            evidence_options = sorted(

                filtered[
                    "Evidence Level"
                ]
                .dropna()
                .unique()
            )


            selected_evidence = (
                st.multiselect(

                    "Evidence level",

                    evidence_options,

                    default=
                        evidence_options
                )
            )


            filtered = filtered[

                filtered[
                    "Evidence Level"
                ].isin(
                    selected_evidence
                )
            ]


        receptor = st.selectbox(

            "Related receptor",

            [

                "All",

                "GPR61",

                "GPR135",

                "MT2 receptor",

                "5-HT6 receptor"
            ]
        )


        if (
            receptor != "All"
            and
            "Related Receptors"
            in filtered.columns
        ):

            filtered = filtered[

                filtered[
                    "Related Receptors"
                ]
                .astype(str)
                .str.contains(

                    receptor,

                    case=False,

                    na=False
                )
            ]


        min_ml = st.slider(

            "Minimum GPCRLigNet score",

            min_value=0.0,

            max_value=1.0,

            value=0.0,

            step=0.05
        )


        if (
            "GPCRLigNet Score"
            in filtered.columns
        ):

            filtered[
                "GPCRLigNet Score"
            ] = pd.to_numeric(

                filtered[
                    "GPCRLigNet Score"
                ],

                errors="coerce"
            )


            filtered = filtered[

                filtered[
                    "GPCRLigNet Score"
                ]
                >= min_ml
            ]


        display_columns = [

            c for c in [

                "Rank",

                "Molecule",

                "Related Receptors",

                "GPCRLigNet Score",

                "ML Prediction",

                "Evidence Level",

                "Experimental Validation",

                "Priority",

                "Priority Score",

                "PMIDs",

                "Years",

                "PubChem CID"

            ]

            if c in
            filtered.columns
        ]


        st.write(
            "Candidates:",
            len(filtered)
        )


        st.dataframe(

            filtered[
                display_columns
            ],

            use_container_width=True,

            hide_index=True
        )


        st.download_button(

            "Download filtered candidates",

            filtered.to_csv(
                index=False
            ),

            "gpr62_filtered_candidates.csv",

            "text/csv"
        )


        # ----------------------------------------------------
        # CANDIDATE DETAIL VIEW
        # ----------------------------------------------------

        if (
            not filtered.empty
            and
            "Molecule"
            in filtered.columns
        ):

            st.divider()

            st.subheader(
                "Candidate Details"
            )


            molecule_options = (

                filtered[
                    "Molecule"
                ]
                .dropna()
                .astype(str)
                .tolist()
            )


            selected_molecule = (
                st.selectbox(

                    "Select molecule",

                    molecule_options
                )
            )


            candidate = filtered[

                filtered[
                    "Molecule"
                ].astype(str)

                ==

                selected_molecule

            ].iloc[0]


            left, right = (
                st.columns(
                    [1, 2]
                )
            )


            with left:

                smiles = (
                    candidate.get(
                        "SMILES"
                    )
                )


                image = (
                    smiles_to_image(
                        smiles
                    )
                )


                if image is not None:

                    st.image(
                        image,
                        caption=
                            selected_molecule
                    )


                if valid_value(
                    smiles
                ):

                    st.caption(
                        "SMILES"
                    )

                    st.code(
                        str(smiles),
                        language="text"
                    )


            with right:

                st.markdown(
                    f"## {selected_molecule}"
                )


                score = safe_float(

                    candidate.get(
                        "GPCRLigNet Score"
                    )
                )


                if score is not None:

                    st.metric(
                        "GPCRLigNet Activity Score",
                        f"{score:.4f}"
                    )


                st.write(
                    "**Related receptor(s):**",
                    safe_text(
                        candidate.get(
                            "Related Receptors"
                        )
                    )
                )


                st.write(
                    "**Experimental validation:**",
                    safe_text(
                        candidate.get(
                            "Experimental Validation"
                        )
                    )
                )


                st.write(
                    "**Evidence level:**",
                    safe_text(
                        candidate.get(
                            "Evidence Level"
                        )
                    )
                )


                st.write(
                    "**Priority:**",
                    safe_text(
                        candidate.get(
                            "Priority"
                        )
                    )
                )


                st.write(
                    "**PubChem CID:**",
                    safe_text(
                        candidate.get(
                            "PubChem CID"
                        )
                    )
                )


            evidence_sentence = (
                candidate.get(
                    "Evidence Sentence"
                )
            )


            if valid_value(
                evidence_sentence
            ):

                st.subheader(
                    "Literature Evidence"
                )

                st.info(
                    str(
                        evidence_sentence
                    )
                )


            pmids = (
                candidate.get(
                    "PMIDs"
                )
            )


            if valid_value(pmids):

                st.subheader(
                    "PubMed Evidence"
                )

                show_pmid_links(
                    pmids
                )


# ============================================================
# PREDICT NEW MOLECULE
# ============================================================

elif page == (
    "Predict New Molecule"
):

    st.header(
        "Predict a New Molecule"
    )


    st.write(
        "Enter a SMILES string to generate the same "
        "1024-bit radius-4 Morgan fingerprint used by "
        "GPCRLigNet and obtain a live activity prediction."
    )


    st.warning(
        "This activity score predicts general GPCR-related "
        "activity. It does not directly predict GPR62 binding."
    )


    smiles_input = st.text_input(

        "SMILES",

        placeholder=(
            "Example: "
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
        )
    )


    predict_button = (
        st.button(
            "Run GPCRLigNet Prediction"
        )
    )


    if predict_button:

        if not (
            smiles_input.strip()
        ):

            st.error(
                "Please enter a SMILES string."
            )

        else:

            with st.spinner(
                "Running GPCRLigNet..."
            ):

                result = (
                    predict_gpcr_activity(
                        smiles_input.strip()
                    )
                )


            if not result[
                "success"
            ]:

                st.error(
                    result[
                        "error"
                    ]
                )

            else:

                st.success(
                    "Prediction complete."
                )


                activity = result[
                    "activity_score"
                ]


                inactive = result[
                    "inactive_score"
                ]


                left, right = (
                    st.columns(
                        [1, 2]
                    )
                )


                with left:

                    image = (
                        smiles_to_image(
                            smiles_input
                        )
                    )


                    if image is not None:

                        st.image(
                            image,
                            caption=
                                "Input molecule"
                        )


                with right:

                    st.metric(
                        "GPCRLigNet Activity Score",
                        f"{activity:.4f}"
                    )


                    st.metric(
                        "Inactive Probability",
                        f"{inactive:.4f}"
                    )


                    st.subheader(
                        activity_label(
                            activity
                        )
                    )


                with st.expander(
                    "Model details"
                ):

                    st.write(
                        "Raw model output:",
                        result[
                            "raw_prediction"
                        ]
                    )


                    st.write(
                        "Fingerprint:",
                        "Morgan radius 4"
                    )


                    st.write(
                        "Fingerprint size:",
                        "1024 bits"
                    )


                    st.write(
                        "Model input:",
                        "(1, 1024)"
                    )


                    st.write(
                        "Activity output:",
                        "prediction[0][0]"
                    )


                st.subheader(
                    "SMILES"
                )


                st.code(
                    smiles_input,
                    language="text"
                )


                # ============================================
                # PROJECT DATABASE LOOKUP
                # ============================================

                st.divider()

                st.header(
                    "Existing Project Evidence"
                )


                existing = (
                    find_existing_molecule(
                        smiles_input
                    )
                )


                if existing is None:

                    st.info(
                        "This molecule was not found in the "
                        "current project candidate database. "
                        "The GPCRLigNet score above is therefore "
                        "a new computational prediction without "
                        "a matched literature record in this dataset."
                    )

                else:

                    molecule_name = (
                        safe_text(
                            existing.get(
                                "Molecule"
                            ),
                            "Unknown molecule"
                        )
                    )


                    st.success(
                        "This molecule already exists in "
                        "the project database."
                    )


                    st.markdown(
                        f"## {molecule_name}"
                    )


                    col1, col2, col3 = (
                        st.columns(3)
                    )


                    col1.metric(
                        "Evidence Level",
                        safe_text(
                            existing.get(
                                "Evidence Level"
                            )
                        )
                    )


                    col2.metric(
                        "Validation",
                        safe_text(
                            existing.get(
                                "Experimental Validation"
                            )
                        )
                    )


                    priority = (
                        safe_float(
                            existing.get(
                                "Priority Score"
                            )
                        )
                    )


                    if priority is None:

                        priority_display = (
                            "N/A"
                        )

                    else:

                        priority_display = (
                            f"{priority:.3f}"
                        )


                    col3.metric(
                        "Priority Score",
                        priority_display
                    )


                    st.write(
                        "**Related receptor(s):**",
                        safe_text(
                            existing.get(
                                "Related Receptors"
                            )
                        )
                    )


                    st.write(
                        "**Candidate type:**",
                        safe_text(
                            existing.get(
                                "Candidate Type"
                            )
                        )
                    )


                    st.write(
                        "**Entity class:**",
                        safe_text(
                            existing.get(
                                "Entity Class"
                            )
                        )
                    )


                    st.write(
                        "**Literature score:**",
                        safe_text(
                            existing.get(
                                "Literature Score"
                            )
                        )
                    )


                    st.write(
                        "**Receptor relevance:**",
                        safe_text(
                            existing.get(
                                "Receptor Relevance"
                            )
                        )
                    )


                    # ----------------------------------------
                    # GPR62 VS RELATED RECEPTORS
                    # ----------------------------------------

                    st.subheader(
                        "GPR62 vs Related Receptors"
                    )


                    receptor_flags = (
                        pd.DataFrame({

                            "Receptor": [

                                "GPR62",

                                "GPR61",

                                "GPR135",

                                "MT2",

                                "5-HT6"
                            ],

                            "Evidence in Project": [

                                safe_text(
                                    existing.get(
                                        "GPR62"
                                    ),
                                    ""
                                ),

                                safe_text(
                                    existing.get(
                                        "GPR61"
                                    ),
                                    ""
                                ),

                                safe_text(
                                    existing.get(
                                        "GPR135"
                                    ),
                                    ""
                                ),

                                safe_text(
                                    existing.get(
                                        "MT2"
                                    ),
                                    ""
                                ),

                                safe_text(
                                    existing.get(
                                        "5-HT6"
                                    ),
                                    ""
                                )
                            ]
                        })
                    )


                    st.dataframe(

                        receptor_flags,

                        use_container_width=True,

                        hide_index=True
                    )


                    evidence_sentence = (
                        existing.get(
                            "Evidence Sentence"
                        )
                    )


                    if valid_value(
                        evidence_sentence
                    ):

                        st.subheader(
                            "Literature Evidence"
                        )

                        st.info(
                            str(
                                evidence_sentence
                            )
                        )


                    pmids = (
                        existing.get(
                            "PMIDs"
                        )
                    )


                    if valid_value(
                        pmids
                    ):

                        st.subheader(
                            "PubMed / Literature Sources"
                        )

                        show_pmid_links(
                            pmids
                        )


                    years = (
                        existing.get(
                            "Years"
                        )
                    )


                    if valid_value(
                        years
                    ):

                        st.write(
                            "**Publication year(s):**",
                            years
                        )


                    article_titles = (
                        existing.get(
                            "Article Titles"
                        )
                    )


                    if valid_value(
                        article_titles
                    ):

                        st.subheader(
                            "Related Articles"
                        )


                        for article_title in (

                            str(
                                article_titles
                            )
                            .split("|")

                        ):

                            if (
                                article_title
                                .strip()
                            ):

                                st.write(
                                    "•",
                                    article_title
                                    .strip()
                                )


                    # ----------------------------------------
                    # PAPER LEVEL EVIDENCE
                    # ----------------------------------------

                    evidence_rows = (
                        find_master_evidence(
                            molecule_name
                        )
                    )


                    if not (
                        evidence_rows.empty
                    ):

                        st.subheader(
                            "Paper-Level Evidence"
                        )


                        evidence_columns = [

                            c for c in [

                                "Related Receptor",

                                "Database",

                                "Year",

                                "PMID",

                                "PMCID",

                                "Journal",

                                "Article Title",

                                "Experimental Validation",

                                "Evidence Level",

                                "Evidence Sentence",

                                "PubMed Link",

                                "PMC Link"

                            ]

                            if c in
                            evidence_rows.columns
                        ]


                        st.dataframe(

                            evidence_rows[
                                evidence_columns
                            ],

                            use_container_width=True,

                            hide_index=True
                        )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == (
    "Batch Prediction"
):

    st.header(
        "Batch GPCRLigNet Prediction"
    )


    st.write(
        "Upload a CSV containing a column named `SMILES`. "
        "The app will score every valid molecule using "
        "GPCRLigNet and rank the results."
    )


    template = pd.DataFrame({

        "Molecule": [
            "Caffeine"
        ],

        "SMILES": [
            (
                "Cn1cnc2c1c(=O)"
                "n(C)c(=O)n2C"
            )
        ]
    })


    st.download_button(

        "Download CSV template",

        template.to_csv(
            index=False
        ),

        "gpcrlignet_batch_template.csv",

        "text/csv"
    )


    uploaded = (
        st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )
    )


    if uploaded is not None:

        try:

            batch_df = (
                pd.read_csv(
                    uploaded
                )
            )

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )

            batch_df = (
                pd.DataFrame()
            )


        if not batch_df.empty:

            if (
                "SMILES"
                not in batch_df.columns
            ):

                st.error(
                    "The uploaded CSV must contain "
                    "a column named SMILES."
                )

            else:

                st.write(
                    "Input molecules:",
                    len(batch_df)
                )


                if st.button(
                    "Run Batch Prediction"
                ):

                    model = (
                        load_gpcrlignet()
                    )


                    results = []


                    progress = (
                        st.progress(0)
                    )


                    total = len(
                        batch_df
                    )


                    for i, row in (
                        batch_df.iterrows()
                    ):

                        smiles = (
                            row.get(
                                "SMILES"
                            )
                        )


                        molecule_name = (
                            row.get(
                                "Molecule",
                                f"Molecule {i + 1}"
                            )
                        )


                        result = (
                            predict_gpcr_activity(
                                smiles
                            )
                        )


                        if result[
                            "success"
                        ]:

                            score = (
                                result[
                                    "activity_score"
                                ]
                            )


                            existing = (
                                find_existing_molecule(
                                    smiles
                                )
                            )


                            if (
                                existing
                                is not None
                            ):

                                related = (
                                    safe_text(
                                        existing.get(
                                            "Related Receptors"
                                        )
                                    )
                                )

                                evidence = (
                                    safe_text(
                                        existing.get(
                                            "Evidence Level"
                                        )
                                    )
                                )

                                validation = (
                                    safe_text(
                                        existing.get(
                                            "Experimental Validation"
                                        )
                                    )
                                )

                                database_match = (
                                    "Yes"
                                )

                            else:

                                related = ""
                                evidence = ""
                                validation = ""
                                database_match = (
                                    "No"
                                )


                            results.append({

                                "Molecule":
                                    molecule_name,

                                "SMILES":
                                    smiles,

                                "GPCRLigNet Score":
                                    score,

                                "Prediction":
                                    activity_label(
                                        score
                                    ),

                                "Database Match":
                                    database_match,

                                "Related Receptors":
                                    related,

                                "Evidence Level":
                                    evidence,

                                "Experimental Validation":
                                    validation
                            })


                        else:

                            results.append({

                                "Molecule":
                                    molecule_name,

                                "SMILES":
                                    smiles,

                                "GPCRLigNet Score":
                                    None,

                                "Prediction":
                                    "Invalid / failed",

                                "Database Match":
                                    "No",

                                "Related Receptors":
                                    "",

                                "Evidence Level":
                                    "",

                                "Experimental Validation":
                                    ""
                            })


                        progress.progress(
                            int(
                                (
                                    i + 1
                                )
                                /
                                total
                                * 100
                            )
                        )


                    results_df = (
                        pd.DataFrame(
                            results
                        )
                    )


                    results_df[
                        "GPCRLigNet Score"
                    ] = pd.to_numeric(

                        results_df[
                            "GPCRLigNet Score"
                        ],

                        errors="coerce"
                    )


                    results_df = (
                        results_df.sort_values(

                            "GPCRLigNet Score",

                            ascending=False,

                            na_position="last"
                        )
                    )


                    results_df.insert(

                        0,

                        "Rank",

                        range(
                            1,
                            len(results_df)
                            + 1
                        )
                    )


                    st.success(
                        "Batch prediction complete."
                    )


                    st.dataframe(

                        results_df,

                        use_container_width=True,

                        hide_index=True
                    )


                    st.download_button(

                        "Download ranked batch predictions",

                        results_df.to_csv(
                            index=False
                        ),

                        "gpcrlignet_batch_predictions.csv",

                        "text/csv"
                    )


# ============================================================
# FULL EVIDENCE DATABASE
# ============================================================

elif page == (
    "Full Evidence Database"
):

    st.header(
        "Full Literature Evidence Database"
    )


    st.write(
        "Each row represents a molecule–paper relationship. "
        "Use this page to trace candidate molecules back "
        "to individual literature records."
    )


    if master.empty:

        st.error(
            "Enriched master dataset was not found."
        )

    else:

        filtered = (
            master.copy()
        )


        search = st.text_input(
            "Search entire evidence database"
        )


        if search:

            mask = (

                filtered
                .astype(str)
                .apply(

                    lambda row:

                    row.str.contains(

                        search,

                        case=False,

                        na=False

                    ).any(),

                    axis=1
                )
            )


            filtered = (
                filtered[mask]
            )


        if (
            "Related Receptor"
            in filtered.columns
        ):

            receptor_options = sorted(

                filtered[
                    "Related Receptor"
                ]
                .dropna()
                .unique()
            )


            selected_receptors = (
                st.multiselect(

                    "Related receptors",

                    receptor_options,

                    default=
                        receptor_options
                )
            )


            filtered = filtered[

                filtered[
                    "Related Receptor"
                ].isin(
                    selected_receptors
                )
            ]


        if (
            "Experimental Validation"
            in filtered.columns
        ):

            validation_options = sorted(

                filtered[
                    "Experimental Validation"
                ]
                .dropna()
                .unique()
            )


            selected_validation = (
                st.multiselect(

                    "Experimental validation",

                    validation_options,

                    default=
                        validation_options
                )
            )


            filtered = filtered[

                filtered[
                    "Experimental Validation"
                ].isin(
                    selected_validation
                )
            ]


        if (
            "Evidence Level"
            in filtered.columns
        ):

            evidence_options = sorted(

                filtered[
                    "Evidence Level"
                ]
                .dropna()
                .unique()
            )


            selected_evidence = (
                st.multiselect(

                    "Evidence level",

                    evidence_options,

                    default=
                        evidence_options
                )
            )


            filtered = filtered[

                filtered[
                    "Evidence Level"
                ].isin(
                    selected_evidence
                )
            ]


        st.write(
            "Rows:",
            len(filtered)
        )


        st.dataframe(

            filtered,

            use_container_width=True,

            hide_index=True
        )


        st.download_button(

            "Download filtered evidence database",

            filtered.to_csv(
                index=False
            ),

            "gpr62_full_evidence_filtered.csv",

            "text/csv"
        )