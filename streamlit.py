```python
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Alzheimer's ML Research",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* MAIN APP */

.stApp {
    background-color: #f4f7fb;
    color: #1a1a2e;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 2px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* HEADERS */

h1, h2, h3, h4, h5, h6 {
    color: #1a1a2e !important;
    font-weight: 700 !important;
}

/* TEXT */

p, li, div, span, label {
    color: #1a1a2e !important;
}

/* HERO */

.hero-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 800;
    color: #1a1a2e !important;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #0f7173 !important;
    margin-bottom: 2rem;
}

/* CARDS */

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid #dbe4ee;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.04);
}

/* ABSTRACT */

.abstract-box {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    border-left: 6px solid #0f7173;
    border: 1px solid #dbe4ee;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.04);
}

/* METRICS */

div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #dbe4ee;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.04);
}

div[data-testid="metric-container"] label {
    color: #0f7173 !important;
    font-weight: 700;
}

div[data-testid="metric-container"] div {
    color: #1a1a2e !important;
}

/* ALERTS */

.stAlert {
    border-radius: 16px !important;
    border: 1px solid #dbe4ee !important;
}

/* TABLES */

table {
    border-collapse: collapse !important;
    width: 100% !important;
    background: white !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #dbe4ee !important;
}

thead tr {
    background-color: #0f7173 !important;
}

thead th {
    color: white !important;
    padding: 14px !important;
    border: 1px solid #dbe4ee !important;
}

tbody td {
    color: #1a1a2e !important;
    background-color: white !important;
    border: 1px solid #dbe4ee !important;
    padding: 12px !important;
}

/* DATAFRAMES */

[data-testid="stDataFrame"] {
    border: 1px solid #dbe4ee !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* TABS */

.stTabs [data-baseweb="tab"] {
    background: white !important;
    border: 1px solid #dbe4ee !important;
    border-radius: 10px !important;
    margin-right: 8px;
    padding: 10px 16px !important;
}

.stTabs [aria-selected="true"] {
    background: #0f7173 !important;
    color: white !important;
}

/* EXPANDERS */

details {
    background: white !important;
    border: 1px solid #dbe4ee !important;
    border-radius: 16px !important;
    padding: 8px !important;
}

/* CODE BLOCKS */

pre {
    background-color: #111827 !important;
    color: #f8fafc !important;
    border-radius: 16px !important;
    padding: 16px !important;
    border: 1px solid #374151 !important;
}

code {
    color: #f8fafc !important;
    background-color: #111827 !important;
}

div[data-testid="stCodeBlock"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* DROPDOWNS */

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #1a1a2e !important;
    border: 1px solid #dbe4ee !important;
}

/* PIPELINE */

.pipeline-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-top: 20px;
}

.pipeline-step {
    background: white;
    border-left: 8px solid #0f7173;
    border-radius: 18px;
    padding: 20px;
    width: 85%;
    text-align: center;
    font-weight: 700;
    color: #1a1a2e !important;
    border: 1px solid #dbe4ee;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.04);
}

.pipeline-arrow {
    font-size: 28px;
    color: #0f7173 !important;
    font-weight: bold;
}

.takeaway-box {
    background: white;
    border-left: 6px solid #0f7173;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    font-size: 1.2rem;
    border: 1px solid #dbe4ee;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.04);
}

</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.markdown("""
# Alzheimer's ML Research
### Sumeet Kapoor
*USC • Multimodal Healthcare AI*
""")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview & Motivation",
        "Dataset & Feature Design",
        "Preprocessing Pipeline",
        "Model Selection & Baseline Results",
        "Missing Modality Experiment",
        "Robustness Training & Cross-Validation",
        "Ablation Study & Final Conclusions"
    ]
)

# =========================================
# DATA
# =========================================

model_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Extra Trees"],
    "ROC-AUC": [0.977, 0.968, 0.965]
})

modality_df = pd.DataFrame({
    "Model Type": ["Clinical Only", "MRI Only", "Multimodal"],
    "ROC-AUC": [0.969, 0.825, 0.977],
    "Features": [6, 30, 36]
})

ablation_df = pd.DataFrame({
    "Dropout Rate": [0, 10, 30, 50],
    "Full AUC": [0.977, 0.977, 0.975, 0.969],
    "Missing AUC": [0.771, 0.817, 0.792, 0.827],
    "Performance Drop": [0.205, 0.160, 0.183, 0.142]
})

# =========================================
# PAGE 1
# =========================================

if page == "Overview & Motivation":

    st.markdown('<div class="hero-title">Multimodal Alzheimer\'s Disease Classification Under Missing Clinical Data</div>', unsafe_allow_html=True)

    st.markdown('<div class="hero-subtitle">Investigating robustness of multimodal ML systems in incomplete real-world healthcare settings</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="abstract-box">
    This project investigates how multimodal machine learning systems for Alzheimer's disease diagnosis behave when real-world clinical data becomes incomplete. Rather than optimizing only for benchmark accuracy, this work focuses on clinical deployability and robustness under missing modalities.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Baseline ROC-AUC", "0.977")

    with c2:
        st.metric("Performance Drop (no robustness)", "-0.205")

    with c3:
        st.metric("Performance Drop (with robustness)", "-0.114")

# =========================================
# PAGE 2
# =========================================

elif page == "Dataset & Feature Design":

    st.title("Dataset & Feature Design")

    tab1, tab2 = st.tabs(["Clinical Modality", "MRI Modality"])

    with tab1:

        clinical_df = pd.DataFrame({
            "Feature": ["AGE", "SEX", "EDUC", "RAVLTFG", "RAVLTIMM", "TRABSCOR"],
            "Meaning": [
                "Age risk",
                "Sex prevalence",
                "Cognitive reserve",
                "Delayed recall",
                "Immediate recall",
                "Executive function"
            ]
        })

        st.table(clinical_df)

    with tab2:

        st.info("""
        **Final MRI Representation:**  
        Top 30 variance-selected structural neuroimaging biomarkers.
        """)

# =========================================
# PAGE 3
# =========================================

elif page == "Preprocessing Pipeline":

    st.title("End-to-End Data Pipeline")

    pipeline_steps = [
        "1. .rda → CSV Conversion",
        "2. Diagnosis Filtering (CN / DEM only)",
        "3. Patient-Level Merging (Clinical + MRI on RID)",
        "4. MRI Deduplication (latest scan per patient)",
        "5. High-Missing Feature Removal (>30% threshold)",
        "6. Variance-Based MRI Feature Selection (Top 30)",
        "7. Median Imputation",
        "8. Stratified 80/20 Train/Test Split",
        "9. StandardScaler Normalization"
    ]

    pipeline_html = '<div class="pipeline-container">'

    for i, step in enumerate(pipeline_steps):

        pipeline_html += f'''
        <div class="pipeline-step">
            {step}
        </div>
        '''

        if i != len(pipeline_steps) - 1:
            pipeline_html += '''
            <div class="pipeline-arrow">↓</div>
            '''

    pipeline_html += '</div>'

    st.markdown(pipeline_html, unsafe_allow_html=True)

# =========================================
# PAGE 4
# =========================================

elif page == "Model Selection & Baseline Results":

    st.title("Model Selection & Baseline Results")

    fig = px.bar(
        model_results,
        x="Model",
        y="ROC-AUC",
        color="Model"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a2e"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("ROC Curves")

    roc_df = pd.DataFrame({
        "FPR": [0, 0.05, 0.1, 0.2, 1],
        "LR": [0, 0.75, 0.90, 0.97, 1],
        "RF": [0, 0.70, 0.87, 0.94, 1],
        "ET": [0, 0.68, 0.85, 0.92, 1]
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=roc_df["FPR"],
        y=roc_df["LR"],
        mode="lines",
        name="Logistic Regression (0.977)"
    ))

    fig.add_trace(go.Scatter(
        x=roc_df["FPR"],
        y=roc_df["RF"],
        mode="lines",
        name="Random Forest (0.968)"
    ))

    fig.add_trace(go.Scatter(
        x=roc_df["FPR"],
        y=roc_df["ET"],
        mode="lines",
        name="Extra Trees (0.965)"
    ))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a2e"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================
# PAGE 5
# =========================================

elif page == "Missing Modality Experiment":

    st.title("Missing Modality Experiment")

    st.warning("""
    Real-world healthcare systems frequently contain incomplete clinical records.
    """)

    st.subheader("Simulation Method")

    st.code("""
X_test_missing[clinical_features] = 0
    """, language="python")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Full Multimodal AUC", "0.977")

    with c2:
        st.metric("Missing Clinical AUC", "0.771", delta="-0.205")

    st.error("""
    A drop of 0.205 ROC-AUC represents major degradation in diagnostic performance.
    """)

    st.divider()

    st.subheader("Modality Comparison")

    st.table(modality_df)

# =========================================
# PAGE 6
# =========================================

elif page == "Robustness Training & Cross-Validation":

    st.title("Robustness Training & Cross-Validation")

    missingness_df = pd.DataFrame({
        "Missingness": [10, 30, 50, 70],
        "Baseline AUC": [0.919, 0.879, 0.820, 0.816],
        "Robust AUC": [0.987, 0.988, 0.980, 0.948]
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=missingness_df["Missingness"],
        y=missingness_df["Baseline AUC"],
        mode='lines+markers',
        name='Baseline'
    ))

    fig.add_trace(go.Scatter(
        x=missingness_df["Missingness"],
        y=missingness_df["Robust AUC"],
        mode='lines+markers',
        name='Robust'
    ))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a2e",
        xaxis_title="% Missing Clinical Data",
        yaxis_title="ROC-AUC"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================
# PAGE 7
# =========================================

elif page == "Ablation Study & Final Conclusions":

    st.title("Ablation Study & Final Conclusions")

    st.table(ablation_df)

    fig = px.line(
        ablation_df,
        x="Dropout Rate",
        y="Missing AUC",
        markers=True
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1a1a2e"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="takeaway-box">
    This project shifts the focus from maximizing ideal-condition accuracy to designing clinically deployable healthcare AI systems.
    </div>
    """, unsafe_allow_html=True)
```
