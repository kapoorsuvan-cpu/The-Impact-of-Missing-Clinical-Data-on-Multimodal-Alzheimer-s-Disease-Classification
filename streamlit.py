import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Alzheimer's Research Demo",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# STYLING
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

[data-testid="stSidebar"] {
    background-color: #020617;
}

.block-container {
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #111827 100%);
    padding: 4rem;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 60px;
    font-weight: 800;
    line-height: 1.05;
    color: white;
    margin-bottom: 1.5rem;
}

.hero-text {
    font-size: 20px;
    color: #cbd5e1;
    line-height: 1.9;
    max-width: 1050px;
}

.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1.5rem;
    border-radius: 18px;
}

.metric-title {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: white;
    font-size: 32px;
    font-weight: 800;
    margin-top: 0.5rem;
}

.section-title {
    font-size: 40px;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.context-box {
    background-color: #f8fafc;
    border-left: 6px solid #2563eb;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
Multimodal Alzheimer's Disease Classification Under Missing Clinical Data
</div>

<div class="hero-text">
This dashboard demonstrates the machine learning pipeline, robustness experiments, and research findings developed for a multimodal Alzheimer's disease classification project using the ADNI dataset.
<br><br>
The project investigates a central translational healthcare AI problem: modern multimodal models often assume perfectly complete patient records, but real clinical environments frequently contain incomplete cognitive testing, unavailable imaging studies, and fragmented electronic health records.
<br><br>
Rather than focusing only on benchmark accuracy, this research evaluates how multimodal systems behave under missing clinical information and explores whether robustness-oriented training strategies can improve resilience under realistic deployment conditions.
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-title">Best ROC-AUC</div>
        <div class="metric-value">0.977</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-title">Missing Modality AUC</div>
        <div class="metric-value">0.771</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-title">Robust Model AUC</div>
        <div class="metric-value">0.863</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-title">Core Contribution</div>
        <div class="metric-value">Robust Healthcare AI</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Research Walkthrough")

section = st.sidebar.radio(
    "Pipeline Sections",
    [
        "1. Research Motivation",
        "2. Dataset Construction",
        "3. Clinical Feature Selection",
        "4. MRI Biomarker Processing",
        "5. Machine Learning Pipeline",
        "6. Baseline Model Results",
        "7. Missing-Modality Experiment",
        "8. Robustness Training",
        "9. Missingness Analysis",
        "10. Key Findings & Future Work"
    ]
)

# =========================================================
# SECTION 1
# =========================================================

if section == "1. Research Motivation":

    st.markdown('<div class="section-title">Research Motivation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="context-box">
    Most multimodal healthcare AI systems are developed and evaluated under idealized conditions where all patient information is assumed to be complete and available.
    <br><br>
    However, real clinical environments rarely operate under these conditions. Cognitive testing may be incomplete, demographic variables may be missing, MRI scans may be unavailable, and patient records may be fragmented across healthcare systems.
    <br><br>
    This project investigates how multimodal Alzheimer's disease classification systems degrade under missing clinical information and whether robustness-oriented training strategies can improve resilience under incomplete healthcare data conditions.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Research Question")

    st.success("""
    Can multimodal Alzheimer's disease classification systems remain clinically useful when cognitive and clinical information becomes incomplete or unavailable?
    """)

# =========================================================
# SECTION 2
# =========================================================

elif section == "2. Dataset Construction":

    st.markdown('<div class="section-title">Dataset Construction</div>', unsafe_allow_html=True)

    st.write("""
    The project uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), one of the largest and most widely used public Alzheimer's research datasets.
    """)

    dataset_df = pd.DataFrame({
        "Dataset Components": [
            "Demographic Information",
            "Cognitive Testing",
            "MRI Imaging",
            "PET Imaging",
            "Longitudinal Follow-Up",
            "Clinical Diagnoses"
        ]
    })

    st.dataframe(dataset_df, use_container_width=True)

    st.info("""
    The project focused specifically on binary classification between cognitively normal (CN) patients and dementia (DEM) patients.
    """)

# =========================================================
# SECTION 3
# =========================================================

elif section == "3. Clinical Feature Selection":

    st.markdown('<div class="section-title">Clinical Feature Selection</div>', unsafe_allow_html=True)

    st.write("""
    Clinical and cognitive variables were selected based on biological relevance rather than purely predictive strength.
    """)

    clinical_df = pd.DataFrame({
        "Feature": [
            "AGE",
            "SEX",
            "EDUC",
            "RAVLTFG",
            "RAVLTIMM",
            "TRABSCOR"
        ],
        "Purpose": [
            "Age-related disease risk",
            "Sex-related prevalence differences",
            "Cognitive reserve",
            "Delayed verbal recall",
            "Immediate verbal recall",
            "Executive function"
        ]
    })

    st.dataframe(clinical_df, use_container_width=True)

    st.warning("""
    Highly diagnostic variables such as MMSE, MOCA, and CDRSB were intentionally removed to avoid semantic leakage and artificially inflated model performance.
    """)

# =========================================================
# SECTION 4
# =========================================================

elif section == "4. MRI Biomarker Processing":

    st.markdown('<div class="section-title">MRI Biomarker Processing</div>', unsafe_allow_html=True)

    st.write("""
    Structural MRI biomarkers were extracted from the UCSFFSX6 FreeSurfer dataset.
    """)

    st.markdown("""
    ### MRI Processing Workflow
    - Removed metadata variables
    - Filtered highly missing biomarkers
    - Retained only numeric MRI features
    - Applied variance filtering
    - Selected top 30 MRI biomarkers
    - Performed median imputation for missing values
    - Applied feature standardization
    """)

    st.success("""
    MRI biomarkers provided complementary anatomical information beyond cognitive testing and remained predictive even when clinical variables became unavailable.
    """)

# =========================================================
# SECTION 5
# =========================================================

elif section == "5. Machine Learning Pipeline":

    st.markdown('<div class="section-title">Machine Learning Pipeline</div>', unsafe_allow_html=True)

    pipeline_df = pd.DataFrame({
        "Pipeline Stage": [
            "Train/Test Split",
            "Feature Scaling",
            "Missing Value Handling",
            "Cross-Validation",
            "Model Training",
            "Robustness Evaluation"
        ],
        "Method": [
            "Stratified Split",
            "StandardScaler",
            "Median Imputation",
            "5-Fold Stratified CV",
            "Logistic Regression / RF / Extra Trees",
            "Missing-Modality Simulation"
        ]
    })

    st.dataframe(pipeline_df, use_container_width=True)

    st.info("""
    Logistic Regression ultimately emerged as the strongest and most stable model after multimodal integration and preprocessing.
    """)

# =========================================================
# SECTION 6
# =========================================================

elif section == "6. Baseline Model Results":

    st.markdown('<div class="section-title">Baseline Model Results</div>', unsafe_allow_html=True)

    model_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Extra Trees"
        ],
        "ROC-AUC": [
            0.977,
            0.952,
            0.961
        ]
    })

    fig = px.bar(
        model_df,
        x="Model",
        y="ROC-AUC",
        text="ROC-AUC",
        title="Baseline Model Comparison"
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    The baseline multimodal model achieved strong discriminative capability, with Logistic Regression producing the strongest overall performance.
    """)

# =========================================================
# SECTION 7
# =========================================================

elif section == "7. Missing-Modality Experiment":

    st.markdown('<div class="section-title">Missing-Modality Experiment</div>', unsafe_allow_html=True)

    st.write("""
    To simulate realistic healthcare deployment conditions, all clinical variables were removed from the test data while MRI biomarkers remained available.
    """)

    comparison_df = pd.DataFrame({
        "Condition": [
            "Full Multimodal Data",
            "Missing Clinical Modality"
        ],
        "ROC-AUC": [
            0.977,
            0.771
        ]
    })

    fig = px.bar(
        comparison_df,
        x="Condition",
        y="ROC-AUC",
        text="ROC-AUC",
        title="Performance Under Missing Clinical Information"
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    st.error("""
    Removing clinical information caused a substantial performance decline, demonstrating that standard multimodal systems can become highly dependent on cognitive features.
    """)

# =========================================================
# SECTION 8
# =========================================================

elif section == "8. Robustness Training":

    st.markdown('<div class="section-title">Robustness Training Through Modality Dropout</div>', unsafe_allow_html=True)

    st.write("""
    To improve resilience under incomplete healthcare data conditions, modality dropout training was introduced.
    """)

    st.markdown("""
    ### Training Strategy
    During training:
    - subsets of patients had clinical variables randomly removed,
    - MRI biomarkers remained available,
    - and the model was forced to learn distributed multimodal representations instead of over-relying on clinical information.
    """)

    robust_df = pd.DataFrame({
        "Model": ["Baseline", "Robust"],
        "Full Data AUC": [0.980, 0.977],
        "Missing Data AUC": [0.840, 0.863]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Full Data',
        x=robust_df['Model'],
        y=robust_df['Full Data AUC']
    ))

    fig.add_trace(go.Bar(
        name='Missing Clinical Data',
        x=robust_df['Model'],
        y=robust_df['Missing Data AUC']
    ))

    fig.update_layout(
        barmode='group',
        title='Cross-Validated Robustness Performance'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SECTION 9
# =========================================================

elif section == "9. Missingness Analysis":

    st.markdown('<div class="section-title">Progressive Missingness Analysis</div>', unsafe_allow_html=True)

    missing_df = pd.DataFrame({
        "Missingness": [10, 30, 50, 70],
        "Baseline": [0.919, 0.879, 0.820, 0.816],
        "Robust": [0.987, 0.988, 0.980, 0.948]
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=missing_df["Missingness"],
        y=missing_df["Baseline"],
        mode='lines+markers',
        name='Baseline Model'
    ))

    fig.add_trace(go.Scatter(
        x=missing_df["Missingness"],
        y=missing_df["Robust"],
        mode='lines+markers',
        name='Robust Model'
    ))

    fig.update_layout(
        title="Performance Under Increasing Clinical Missingness",
        xaxis_title="Percentage of Missing Clinical Information",
        yaxis_title="ROC-AUC"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    The robustness-trained model remained substantially more stable as missingness severity increased, demonstrating improved resilience under incomplete healthcare conditions.
    """)

# =========================================================
# SECTION 10
# =========================================================

elif section == "10. Key Findings & Future Work":

    st.markdown('<div class="section-title">Key Findings & Future Work</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Main Findings
    - Standard multimodal systems become vulnerable when clinical information disappears.
    - MRI biomarkers provide meaningful complementary anatomical information.
    - Modality dropout substantially improves robustness under missing clinical data conditions.
    - Robustness improvements can be achieved without major sacrifice in baseline predictive performance.
    """)

    st.markdown("""
    ### Potential Future Directions
    - External validation on OASIS and NACC datasets
    - Longitudinal disease progression modeling
    - Transformer-based multimodal architectures
    - Temporal patient-state modeling
    - Clinically realistic missingness simulations
    - Fairness and demographic robustness analysis
    """)

    st.info("""
    The long-term goal of this research is to help shift healthcare AI from benchmark-oriented evaluation toward clinically deployable systems capable of operating under imperfect real-world conditions.
    """)
