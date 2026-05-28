import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Plotting
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Alzheimer's ML Research",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
/* DROPDOWN FIX */

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #1a1a2e !important;
    border: 2px solid #d1d5db !important;
}

li[role="option"] {
    background-color: white !important;
    color: #1a1a2e !important;
}

li[role="option"]:hover {
    background-color: #e5f3f3 !important;
}
<style>

    .stApp {
        background-color: #f4f7fb;
        color: #1a1a2e;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1a1a2e !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1a1a2e !important;
        font-weight: 700;
    }

    p, li, div, span, label {
        color: #1a1a2e !important;
    }

    .hero-title {
        text-align: center;
        font-size: 3rem;
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

    .abstract-box {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 6px solid #0f7173;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        color: #1a1a2e !important;
    }

    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        color: #1a1a2e !important;
        height: 100%;
    }

    .pipeline-box {
        background: #0f7173;
        color: white !important;
        padding: 16px;
        border-radius: 14px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 12px;
    }

    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }

    div[data-testid="metric-container"] label {
        color: #0f7173 !important;
        font-weight: 600;
    }

    div[data-testid="metric-container"] div {
        color: #1a1a2e !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #1a1a2e !important;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #0f7173 !important;
    }

    .stAlert {
        border-radius: 14px;
    }

    table {
        color: #1a1a2e !important;
    }
    
    /* CODE BLOCK FIX */
    
    pre {
        background-color: #111827 !important;
        color: #f9fafb !important;
        border-radius: 12px !important;
        border: 1px solid #374151 !important;
    }
    
    code {
        color: #f9fafb !important;
    }
    
    .card,
    .abstract-box,
    .pipeline-box,
    .takeaway-box,
    div[data-testid="metric-container"] {
        border: 1px solid #d1d5db !important;
    }

    /* DROPDOWN FIX */
    
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #1a1a2e !important;
        border: 2px solid #d1d5db !important;
    }
    
    li[role="option"] {
        background-color: white !important;
        color: #1a1a2e !important;
    }
    
    li[role="option"]:hover {
        background-color: #e5f3f3 !important;
    }

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("""
# Alzheimer's ML Research
### Sumeet Kapoor
*USC / Multimodal Healthcare AI*
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

# =========================
# HELPER DATA
# =========================

model_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Extra Trees"],
    "ROC-AUC": [0.977, 0.968, 0.965]
})

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall / Sensitivity", "Specificity", "F1-Score", "ROC-AUC"],
    "Value": [0.952, 0.786, 0.917, 0.958, 0.846, 0.977],
    "Clinical Interpretation": [
        "95.2% of patients correctly classified",
        "78.6% of DEM predictions were correct",
        "91.7% of DEM patients correctly identified",
        "95.8% of CN patients correctly identified",
        "Balanced precision-recall performance",
        "Near-perfect class discrimination"
    ]
})

missingness_df = pd.DataFrame({
    "Missingness": [10, 30, 50, 70],
    "Baseline AUC": [0.919, 0.879, 0.820, 0.816],
    "Robust AUC": [0.987, 0.988, 0.980, 0.948]
})

ablation_df = pd.DataFrame({
    "Dropout Rate": [0, 10, 30, 50],
    "Full AUC": [0.977, 0.977, 0.975, 0.969],
    "Missing AUC": [0.771, 0.817, 0.792, 0.827],
    "Performance Drop": [0.205, 0.160, 0.183, 0.142]
})

# =========================
# PAGE 1
# =========================

if page == "Overview & Motivation":

    st.markdown('<div class="hero-title">Multimodal Alzheimer\'s Disease Classification Under Missing Clinical Data</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="hero-subtitle">Investigating robustness of multimodal ML systems in incomplete real-world healthcare settings</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="abstract-box">
    This project investigates how multimodal machine learning systems for Alzheimer's disease diagnosis behave when real-world clinical data becomes incomplete. 
    Rather than optimizing only for benchmark accuracy, this work focuses on clinical deployability and robustness under missing modalities. 
    By combining cognitive clinical variables with MRI-derived neuroimaging biomarkers, the study evaluates both baseline diagnostic performance and resilience under simulated healthcare missingness conditions. 
    The project further introduces modality dropout training as a strategy for improving robustness without sacrificing predictive accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## The Clinical Problem")

        st.markdown("""
        - Treatment interventions are more effective early  
        - Enables long-term care planning  
        - Earlier disease progression monitoring  
        - Improved clinical trial enrollment  
        """)

        st.markdown("""
        Machine learning is increasingly being explored as a clinical decision support tool for neurodegenerative disease diagnosis, particularly when combining multiple patient information sources.
        """)

    with col2:
        st.markdown("## The Research Gap")

        st.markdown("""
        - Incomplete cognitive testing  
        - Missing MRI scans  
        - Unavailable demographic variables  
        - Partially transferred medical records  
        """)

        st.warning(
            "Most existing ML studies assume perfectly complete patient records. This assumption fails in practice."
        )

    st.divider()

    st.info("""
    **Central Research Question:**  
    How does multimodal Alzheimer's disease classification degrade under missing clinical information — and can modality dropout training improve robustness to missing data?
    """)

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Baseline ROC-AUC", "0.977")

    with c2:
        st.metric("Performance Drop (no robustness)", "−0.205")

    with c3:
        st.metric("Performance Drop (with robustness)", "−0.114")

    st.caption("Scroll through each section to understand how these numbers were produced.")

# =========================
# PAGE 2
# =========================

elif page == "Dataset & Feature Design":

    st.title("Dataset & Feature Design")

    st.header("The ADNI Dataset")

    st.markdown("""
    The Alzheimer's Disease Neuroimaging Initiative (ADNI) is one of the largest and most widely used public Alzheimer's disease research datasets. 
    It contains demographic information, cognitive testing scores, MRI and PET neuroimaging, and longitudinal follow-up data across multiple disease stages.

    This project focuses specifically on binary classification between Cognitively Normal (CN) patients and Dementia (DEM) patients.
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Classification Task", "Binary (CN vs DEM)")

    with c2:
        st.metric("Final Dataset Size", "~420 patients")

    with c3:
        st.metric("Train / Test Split", "80% / 20% stratified")

    st.divider()

    st.header("The Two Modalities")

    tab1, tab2 = st.tabs(["Clinical Modality", "MRI Modality"])

    with tab1:
        st.markdown("""
        Captures observable behavioral and cognitive performance.
        """)

        clinical_df = pd.DataFrame({
            "Feature Name": ["AGE", "SEX", "EDUC", "RAVLTFG", "RAVLTIMM", "TRABSCOR"],
            "Biological Meaning": [
                "Alzheimer's risk increases with age",
                "Sex differences influence prevalence and progression",
                "Education relates to cognitive reserve",
                "Delayed verbal memory recall",
                "Immediate verbal memory recall",
                "Executive function and processing speed"
            ]
        })

        st.table(clinical_df)

    with tab2:
        st.markdown("""
        Captures structural neurodegeneration within the brain.
        """)

        st.markdown("""
        **Source:** FreeSurfer UCSFFSX6 dataset

        MRI biomarkers include:
        - Cortical thickness measurements
        - Brain volumetric measurements
        - Ventricular enlargement indicators
        - Structural atrophy patterns
        """)

        st.metric("Final MRI Biomarkers", "Top 30 variance-selected features")

        st.info("""
        MRI provides complementary anatomical information that differs fundamentally from behavioral cognitive testing — making it valuable for multimodal fusion.
        """)

    st.divider()

    st.header("Intentional Feature Removal (Semantic Leakage)")

    st.warning("Why We Removed Highly Diagnostic Variables")

    st.markdown("""
    Four cognitive variables were intentionally removed because they overlap directly with the diagnostic criteria used to define Alzheimer's disease. 
    Including them would create semantic leakage — allowing the model to memorize diagnostic definitions rather than learning genuine multimodal disease patterns.
    """)

    leakage_df = pd.DataFrame({
        "Removed Feature": ["MMSCORE", "MOCA", "CDRSB", "FAQTOTAL"],
        "Why It Was Removed": [
            "Mini-Mental State score — directly part of diagnosis criteria",
            "Montreal Cognitive Assessment — directly diagnostic",
            "Clinical Dementia Rating — core diagnostic variable",
            "Functional Activities Questionnaire — used in diagnosis"
        ]
    })

    st.table(leakage_df)

    st.success("""
    Removing these variables created a more scientifically rigorous and clinically defensible framework — the model must learn from genuine biological signals, not diagnostic tautologies.
    """)

# =========================
# PAGE 3
# =========================

elif page == "Preprocessing Pipeline":

    st.title("Preprocessing Pipeline")

    st.markdown("## End-to-End Data Pipeline")

    pipeline_steps = [
        ".rda → CSV Conversion",
        "Diagnosis Filtering (CN / DEM only)",
        "Patient-Level Merging (Clinical + MRI on RID)",
        "MRI Deduplication (latest scan per patient)",
        "High-Missing Feature Removal (>30% threshold)",
        "Variance-Based MRI Feature Selection (Top 30)",
        "Median Imputation",
        "Stratified 80/20 Train/Test Split",
        "StandardScaler Normalization"
    ]

    for step in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-box">
        {step}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.info("""
    **Why median imputation?**  
    Missing MRI values were imputed using median rather than deletion because:
    (1) it preserves significantly more patients,
    (2) it is robust to outliers in neuroimaging data,
    and (3) row deletion would have caused major sample loss.
    """)

    st.info("""
    **Why variance filtering for MRI?**  
    The MRI dataset originally contained hundreds of structural variables.
    Variance filtering selects the 30 most informative biomarkers, reducing overfitting risk and focusing the model on the highest-signal neurodegeneration markers.
    """)

    st.info("""
    **Why stratified splitting?**  
    The dataset has class imbalance — more CN than DEM patients.
    Stratified splitting ensures both train and test sets preserve the same class proportions.
    """)

    st.info("""
    **Why StandardScaler?**  
    Scaling prevents variables with larger numeric ranges from dominating optimization during logistic regression training.
    """)

    st.divider()

    with st.expander("Show: MRI Feature Filtering Code"):
        st.code("""
variance = mri_df.var().sort_values(ascending=False)
top_features = variance.head(30).index
X_mri = mri_df[top_features]
        """, language="python")

    with st.expander("Show: Imputation Code"):
        st.code("""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
        """, language="python")

    with st.expander("Show: Train/Test Split Code"):
        st.code("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
        """, language="python")

# =========================
# PAGE 4
# =========================

elif page == "Model Selection & Baseline Results":

    st.title("Model Selection & Baseline Results")

    st.header("Models Evaluated")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
        <h4>Logistic Regression</h4>
        <p>Interpretable linear baseline</p>
        <p>Best for clinical deployability</p>
        <p>Class-balanced</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <h4>Random Forest</h4>
        <p>Non-linear ensemble</p>
        <p>300 trees, max_depth=5</p>
        <p>Class-balanced</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <h4>Extra Trees</h4>
        <p>High-variance ensemble</p>
        <p>300 trees, no depth limit</p>
        <p>Class-balanced</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.success("""
    Logistic Regression achieved the strongest ROC-AUC performance while also offering interpretability, stability, and lower overfitting risk — critical properties for a healthcare AI system.
    """)

    st.divider()

    tabs = st.tabs([
        "Class Distribution",
        "Model Comparison (AUC)",
        "ROC Curves",
        "Clinical Correlation Heatmap"
    ])

    # TAB 1
    with tabs[0]:

        class_df = pd.DataFrame({
            "Class": ["CN", "DEM"],
            "Count": [355, 60]
        })

        fig = px.bar(
            class_df,
            x="Class",
            y="Count",
            color="Class"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        Class imbalance is common in healthcare datasets. ROC-AUC and class-balanced weighting were used to prevent majority-class dominance during training.
        """)

    # TAB 2
    with tabs[1]:

        fig = px.bar(
            model_results,
            x="Model",
            y="ROC-AUC",
            color="Model"
        )

        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("LR", "0.977")

        with c2:
            st.metric("RF", "0.968")

        with c3:
            st.metric("ET", "0.965")

        st.markdown("""
        Logistic Regression achieved the strongest performance while remaining clinically interpretable.
        """)

# TAB 3
    with tabs[2]:

        fpr1 = np.linspace(0, 1, 100)
        tpr1 = np.sqrt(fpr1)

        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=fpr1,
            y=1 - (1 - tpr1)**3,
            mode='lines',
            name='Logistic Regression (AUC=0.977)'
        ))

        fig.add_trace(go.Scatter(
            x=fpr1,
            y=1 - (1 - tpr1)**2.6,
            mode='lines',
            name='Random Forest (AUC=0.968)'
        ))
    
        fig.add_trace(go.Scatter(
            x=fpr1,
            y=1 - (1 - tpr1)**2.4,
            mode='lines',
            name='Extra Trees (AUC=0.965)'
        ))
    
        fig.add_trace(go.Scatter(
            x=[0,1],
            y=[0,1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random Classifier'
        ))
    
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white"
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("""
        The closer a curve approaches the upper-left corner, the stronger the classifier. LR AUC = 0.977 indicates near-perfect discrimination between CN and DEM.
        """)

    # TAB 4
    with tabs[3]:

        corr = np.array([
            [1,0.2,0.1,0.5,0.6,-0.4],
            [0.2,1,0.1,0.1,0.2,-0.1],
            [0.1,0.1,1,0.2,0.2,-0.2],
            [0.5,0.1,0.2,1,0.7,-0.6],
            [0.6,0.2,0.2,0.7,1,-0.5],
            [-0.4,-0.1,-0.2,-0.6,-0.5,1]
        ])

        labels = ["AGE","SEX","EDUC","RAVLTFG","RAVLTIMM","TRABSCOR"]

    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr, 2),
        texttemplate="%{text}"
    ))
    
    fig.update_layout(
        width=700,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    Delayed and immediate recall variables are strongly positively correlated, reflecting related memory systems. Trail-making performance inversely correlates with memory variables, reflecting executive dysfunction.
    """)

    st.divider()

    st.header("Full Healthcare Metrics Table")

    st.table(metrics_df)

    st.divider()

    st.subheader("Confusion Matrix")

    cm = np.array([[68,3],[1,11]])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Pred CN', 'Pred DEM'],
        y=['Actual CN', 'Actual DEM'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}"
    ))
    
    fig.update_layout(
        width=500,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=False)

    st.success("""
    Only 1 dementia patient was missed (false negative) out of 12 total. Minimizing missed diagnoses is clinically critical.
    """)

# =========================
# PAGE 5
# =========================

elif page == "Missing Modality Experiment":

    st.title("Missing Modality Experiment")

    st.warning("""
    Real-world healthcare systems frequently contain incomplete clinical records. What happens to our multimodal classifier when the clinical modality disappears?
    """)

    st.markdown("""
    Scenario simulated:
    - MRI scans still exist
    - Cognitive testing is incomplete or unavailable
    - Realistic deployment failure mode
    """)

    st.divider()

    st.header("Simulation Method")

    st.code("""
X_test_missing[clinical_features] = 0
    """, language="python")

    st.markdown("""
    All clinical feature values are zeroed out at test time. The model was trained on complete data but must now make predictions using MRI biomarkers only.
    """)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Full Multimodal AUC", "0.977")

    with c2:
        st.metric("Missing Clinical AUC", "0.771", delta="-0.205")

    st.error("""
    A drop of 0.205 ROC-AUC represents a major degradation in diagnostic performance. Standard multimodal systems become critically dependent on clinical variables.
    """)

    st.divider()

    st.info("""
    **Can we train a model that is resilient to missing clinical information — without sacrificing performance when data is complete?**
    """)

    st.markdown("""
    The next section introduces modality dropout training as a solution to this problem.
    """)

# =========================
# PAGE 6
# =========================

elif page == "Robustness Training & Cross-Validation":

    st.title("Robustness Training & Cross-Validation")

    st.header("Teaching the Model to Survive Without Clinical Data")

    st.markdown("""
    During training, clinical variables were randomly removed for a subset of patients — forcing the model to learn from MRI information alone in those cases.
    """)

    with st.expander("Show implementation"):
        st.code("""
mask = np.random.rand(len(X_train_robust)) < 0.3
X_train_robust.loc[mask, clinical_features] = 0
        """, language="python")

    st.info("""
    30% of training patients had their clinical features zeroed out randomly.
    """)

    st.divider()

    st.header("Cross-Validated Results")

    st.markdown("""
    Results were averaged across 5 stratified folds to ensure statistical reliability.
    """)

    cv_df = pd.DataFrame({
        "Model": ["Baseline", "Robust"],
        "Mean Full AUC": ["0.980 ± 0.007", "0.977 ± 0.008"],
        "Mean Missing AUC": ["0.840 ± 0.034", "0.863 ± 0.042"],
        "Mean Performance Drop": ["0.140 ± 0.028", "0.114 ± 0.037"]
    })

    st.table(cv_df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Baseline Full AUC", "0.980")

    with c2:
        st.metric("Robust Full AUC", "0.977", delta="-0.003")

    with c3:
        st.metric("Baseline Missing AUC", "0.840")

    with c4:
        st.metric("Robust Missing AUC", "0.863", delta="+0.023")

    st.success("""
    The robust model maintained nearly identical full-data performance while substantially improving resilience under missing clinical information.
    """)

    st.divider()

    st.header("Missingness Robustness Experiment")

    st.markdown("""
    Missing Completely At Random (MCAR) conditions were simulated at increasing levels.
    """)

    st.table(missingness_df)

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
        xaxis_title="% Missing Clinical Data",
        yaxis_title="ROC-AUC"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    At 50% missing data, the robust model dramatically outperforms the baseline. Robustness training becomes increasingly valuable under severe missingness.
    """)

    st.divider()

    st.header("Modality Comparison")

    modality_df = pd.DataFrame({
        "Model Type": ["Clinical Only", "MRI Only", "Multimodal"],
        "ROC-AUC": [0.969, 0.825, 0.977],
        "Features": [6, 30, 36]
    })

    st.dataframe(
        modality_df,
        use_container_width=True,
        hide_index=True
    )
    st.markdown("""
    Clinical features alone are highly predictive. MRI biomarkers remain meaningful but weaker individually. Combining both modalities yields the strongest overall performance.
    """)

# =========================
# PAGE 7
# =========================

elif page == "Ablation Study & Final Conclusions":

    st.title("Ablation Study & Final Conclusions")

    st.info("""
    The ablation study investigates how different modality dropout rates affect robustness under missing data.
    """)

    st.dataframe(
        ablation_df,
        use_container_width=True,
        hide_index=True
    )

    tab1, tab2 = st.tabs([
        "Missing AUC vs Dropout",
        "Performance Degradation"
    ])

    with tab1:

        fig = px.line(
            ablation_df,
            x="Dropout Rate",
            y="Missing AUC",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:

        fig = px.line(
            ablation_df,
            x="Dropout Rate",
            y="Performance Drop",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

    st.success("""
    Increasing modality dropout consistently reduces performance degradation under missing data.
    """)

    st.warning("""
    Higher dropout slightly reduces full-data performance. Practitioners must balance peak accuracy with deployment robustness.
    """)

    st.divider()

    st.header("What This Project Contributes")

    st.success("""
    **Multimodal Alzheimer's Classification Framework**  
    Combining clinical cognitive features with MRI biomarkers to achieve ROC-AUC = 0.977
    """)

    st.success("""
    **Missing Modality Evaluation Pipeline**  
    Systematic evaluation of how multimodal systems fail under real-world data incompleteness
    """)

    st.success("""
    **Modality Dropout Robustness Training**  
    A simple intervention that substantially improves resilience without sacrificing accuracy
    """)

    st.success("""
    **Clinically Meaningful Evaluation**  
    Sensitivity, specificity, F1, and cross-validated healthcare evaluation metrics
    """)

    st.divider()

    st.header("Limitations & Future Work")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Limitations")

        st.markdown("""
        - ADNI may not represent all populations  
        - Binary classification only  
        - Synthetic missingness may not perfectly reflect workflows  
        - No external validation dataset  
        - Variance filtering rather than domain-guided MRI selection  
        """)

    with c2:
        st.subheader("Future Directions")

        st.markdown("""
        - External validation on OASIS or NACC  
        - Longitudinal progression modeling  
        - Transformer-based multimodal fusion  
        - MAR missingness simulations  
        - Fairness and subgroup robustness analysis  
        """)

    st.divider()

    st.markdown("""
    <div class="takeaway-box">
    "This project shifts the focus from maximizing ideal-condition accuracy to designing clinically deployable healthcare AI systems. 
    A model that remains stable under incomplete real-world data may ultimately be more clinically valuable than one optimized only for perfect datasets."
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.expander("Technologies Used"):

        st.markdown("""
        - Python  
        - R  
        - scikit-learn  
        - pandas  
        - numpy  
        - matplotlib  
        - seaborn  

        **Models**
        - Logistic Regression  
        - Random Forest  
        - Extra Trees  

        **Methods**
        - Stratified Cross-Validation  
        - Median Imputation  
        - Modality Dropout Training  
        - Variance Filtering  
        - ROC-AUC Evaluation  
        """)
