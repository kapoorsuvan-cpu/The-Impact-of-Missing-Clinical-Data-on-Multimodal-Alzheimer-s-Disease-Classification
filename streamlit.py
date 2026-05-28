# app.py — Streamlit Research Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Multimodal Alzheimer's Disease Robustness Research",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
        margin-bottom: 2rem;
    }

    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    .section-header {
        font-size: 28px;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================
# HERO SECTION
# ============================================
st.markdown(
    """
    <div class="hero">
        <h1>Improving Robustness of Multimodal Alzheimer's Disease Classification Under Missing Clinical Data</h1>
        <p style="font-size:18px;">
        Research project investigating how multimodal machine learning systems behave when clinical information becomes incomplete or unavailable.
        </p>
        <p>
        Focus Areas: Multimodal Healthcare AI • MRI Biomarkers • Missing Modality Robustness • Clinical Deployment
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("Research Dashboard")
section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Research Motivation",
        "Dataset & Modalities",
        "Model Performance",
        "Missing Modality Experiments",
        "Robustness Training",
        "Ablation Study",
        "Clinical Interpretation",
        "Future Work"
    ]
)

# ============================================
# OVERVIEW
# ============================================
if section == "Overview":

    st.header("Project Overview")

    st.write(
        """
        This project investigates how multimodal machine learning systems for Alzheimer's disease diagnosis behave when clinical information becomes incomplete or unavailable.

        While many healthcare AI systems assume perfectly complete patient records, real-world clinical environments frequently contain missing cognitive testing, incomplete medical histories, and unavailable demographic information.

        The goal of this research was therefore not simply to maximize classification accuracy, but to evaluate and improve robustness under missing clinical data conditions.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Best ROC-AUC", "0.977")
    col2.metric("Missing Modality AUC", "0.771")
    col3.metric("Robust Model AUC", "0.863")
    col4.metric("Performance Drop Reduction", "18.6%")

    st.markdown("---")

    st.subheader("Core Research Question")

    st.info(
        "How does multimodal Alzheimer's disease classification degrade under missing clinical information, and can modality dropout training improve robustness to incomplete healthcare data?"
    )

# ============================================
# RESEARCH MOTIVATION
# ============================================
elif section == "Research Motivation":

    st.header("Research Motivation")

    st.write(
        """
        Alzheimer's disease is one of the most important neurodegenerative disorders worldwide. Early diagnosis is clinically valuable because interventions are more effective earlier in disease progression.

        Modern healthcare AI systems increasingly integrate multimodal patient data including:
        """
    )

    modalities = pd.DataFrame({
        "Modality": [
            "Clinical/Cognitive Testing",
            "MRI Imaging",
            "Demographics",
            "Biomarkers",
            "Longitudinal Data"
        ]
    })

    st.dataframe(modalities, use_container_width=True)

    st.warning(
        "Most existing machine learning studies assume complete patient records. Real clinical systems rarely contain fully complete data."
    )

    st.subheader("Examples of Real-World Missingness")

    missingness = pd.DataFrame({
        "Common Clinical Issues": [
            "Incomplete cognitive testing",
            "Unavailable MRI scans",
            "Missing demographic variables",
            "Partial medical record transfers",
            "Inconsistent healthcare documentation"
        ]
    })

    st.dataframe(missingness, use_container_width=True)

# ============================================
# DATASET
# ============================================
elif section == "Dataset & Modalities":

    st.header("Dataset & Modalities")

    st.subheader("ADNI Dataset")

    st.write(
        """
        This project uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), one of the most widely used public Alzheimer's research datasets.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Clinical Features")

        clinical_df = pd.DataFrame({
            "Feature": [
                "AGE",
                "SEX",
                "EDUC",
                "RAVLTFG",
                "RAVLTIMM",
                "TRABSCOR"
            ],
            "Description": [
                "Age",
                "Sex",
                "Education",
                "Delayed verbal recall",
                "Immediate verbal recall",
                "Executive function"
            ]
        })

        st.dataframe(clinical_df, use_container_width=True)

    with col2:
        st.subheader("MRI Biomarkers")

        st.write(
            """
            MRI features were extracted from the UCSFFSX6 FreeSurfer dataset and capture:

            - Cortical thickness
            - Brain volumetrics
            - Ventricular enlargement
            - Regional atrophy patterns
            """
        )

        st.success(
            "MRI biomarkers provide complementary anatomical information beyond cognitive testing."
        )

    st.markdown("---")

    st.subheader("Semantic Leakage Prevention")

    st.write(
        """
        Highly diagnostic variables such as MMSE, MOCA, and CDRSB were intentionally removed to avoid semantic leakage and create a more scientifically defensible modeling framework.
        """
    )

# ============================================
# MODEL PERFORMANCE
# ============================================
elif section == "Model Performance":

    st.header("Baseline Model Performance")

    model_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Extra Trees"
        ],
        "ROC_AUC": [0.977, 0.952, 0.961]
    })

    fig = px.bar(
        model_df,
        x="Model",
        y="ROC_AUC",
        text="ROC_AUC",
        title="Model Comparison"
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Finding")

    st.write(
        """
        Logistic Regression achieved the strongest overall performance while also remaining the most interpretable and stable model.
        """
    )

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Sensitivity",
            "Specificity",
            "ROC-AUC"
        ],
        "Value": [
            0.952,
            0.786,
            0.917,
            0.846,
            0.917,
            0.958,
            0.977
        ]
    })

    st.dataframe(metrics_df, use_container_width=True)

# ============================================
# MISSING MODALITY
# ============================================
elif section == "Missing Modality Experiments":

    st.header("Missing Clinical Modality Experiment")

    st.write(
        """
        To simulate realistic healthcare deployment conditions, all clinical features were removed from the test set while MRI biomarkers remained available.
        """
    )

    comparison_df = pd.DataFrame({
        "Condition": [
            "Full Multimodal Data",
            "Missing Clinical Modality"
        ],
        "ROC_AUC": [0.977, 0.771]
    })

    fig = px.bar(
        comparison_df,
        x="Condition",
        y="ROC_AUC",
        text="ROC_AUC",
        title="Performance Under Missing Clinical Information"
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    st.error(
        "The baseline multimodal model experienced a major performance decline when clinical information disappeared."
    )

    st.metric("Performance Drop", "0.205 ROC-AUC")

# ============================================
# ROBUSTNESS TRAINING
# ============================================
elif section == "Robustness Training":

    st.header("Robustness Training Through Modality Dropout")

    st.write(
        """
        To improve resilience under incomplete healthcare data conditions, modality dropout training was introduced.

        During training, clinical variables were randomly removed for subsets of patients so the model would learn to rely more heavily on MRI information when clinical data became unavailable.
        """
    )

    robust_df = pd.DataFrame({
        "Model": ["Baseline", "Robust"],
        "Full_Data_AUC": [0.980, 0.977],
        "Missing_Data_AUC": [0.840, 0.863]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Full Data',
        x=robust_df['Model'],
        y=robust_df['Full_Data_AUC']
    ))

    fig.add_trace(go.Bar(
        name='Missing Clinical Data',
        x=robust_df['Model'],
        y=robust_df['Missing_Data_AUC']
    ))

    fig.update_layout(
        barmode='group',
        title='Cross-Validated Robustness Performance',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "The robustness-trained model maintained nearly identical full-data performance while substantially improving resilience under missing clinical data conditions."
    )

# ============================================
# ABLATION STUDY
# ============================================
elif section == "Ablation Study":

    st.header("Modality Dropout Ablation Study")

    ablation_df = pd.DataFrame({
        "Dropout_Rate": [0, 10, 30, 50],
        "Missing_AUC": [0.771, 0.817, 0.792, 0.827],
        "Performance_Drop": [0.205, 0.160, 0.183, 0.142]
    })

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(
                            "Missing-Modality Performance",
                            "Performance Degradation"
                        ))

    fig.add_trace(
        go.Scatter(
            x=ablation_df['Dropout_Rate'],
            y=ablation_df['Missing_AUC'],
            mode='lines+markers',
            name='Missing AUC'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=ablation_df['Dropout_Rate'],
            y=ablation_df['Performance_Drop'],
            mode='lines+markers',
            name='Performance Drop'
        ),
        row=1,
        col=2
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        """
        Increasing modality dropout improved robustness under missing clinical data conditions, demonstrating that robustness training distributed predictive importance more effectively across modalities.
        """
    )

# ============================================
# CLINICAL INTERPRETATION
# ============================================
elif section == "Clinical Interpretation":

    st.header("Clinical Interpretation")

    confusion_df = pd.DataFrame({
        "Outcome": [
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives"
        ],
        "Count": [68, 3, 1, 11]
    })

    st.dataframe(confusion_df, use_container_width=True)

    st.write(
        """
        The model demonstrated strong sensitivity and specificity while maintaining a very low false negative count.

        This is clinically important because missed dementia diagnoses may delay treatment, intervention, and long-term patient planning.
        """
    )

    st.info(
        "This research shifts healthcare AI from maximizing ideal-condition accuracy toward building clinically deployable systems robust to imperfect patient records."
    )

# ============================================
# FUTURE WORK
# ============================================
elif section == "Future Work":

    st.header("Future Directions")

    future_work = pd.DataFrame({
        "Future Research Areas": [
            "External validation on OASIS/NACC datasets",
            "Longitudinal disease progression modeling",
            "Transformer-based multimodal systems",
            "Clinically realistic MAR missingness simulations",
            "Fairness and demographic robustness analysis",
            "Temporal patient-state modeling"
        ]
    })

    st.dataframe(future_work, use_container_width=True)

    st.subheader("Technologies Used")

    tech_df = pd.DataFrame({
        "Category": [
            "Languages",
            "Machine Learning",
            "Libraries",
            "Statistical Methods"
        ],
        "Tools": [
            "Python, R",
            "Logistic Regression, Random Forest, Extra Trees",
            "scikit-learn, pandas, NumPy, Plotly",
            "Cross-Validation, Median Imputation, ROC-AUC"
        ]
    })

    st.dataframe(tech_df, use_container_width=True)

    st.markdown("---")

    st.success(
        "This project demonstrates that robustness-focused multimodal healthcare AI may be more clinically valuable than systems optimized only for ideal datasets."
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")

st.caption(
    "Developed by Sumeet Kapoor | Multimodal Healthcare AI Research | Alzheimer's Disease Classification & Missing-Modality Robustness"
)
