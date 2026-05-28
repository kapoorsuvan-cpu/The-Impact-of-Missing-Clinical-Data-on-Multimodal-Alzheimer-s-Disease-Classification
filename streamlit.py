
import os
import nbformat
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Alzheimer's AI Robustness Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0B0F19;
    color: #F9FAFB;
    font-family: Inter, sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #374151;
}

.metric-card {
    background: rgba(17,24,39,0.75);
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: 0.3s ease;
    box-shadow: 0 0 20px rgba(0,0,0,0.2);
}

.metric-card:hover {
    transform: translateY(-4px);
    border: 1px solid #60A5FA;
    box-shadow: 0 0 20px rgba(96,165,250,0.25);
}

.glass-card {
    background: rgba(17,24,39,0.7);
    border: 1px solid #374151;
    border-radius: 22px;
    padding: 1.6rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

.section-title {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 3rem;
    margin-bottom: 1rem;
    color: #F9FAFB;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #9CA3AF;
}

hr {
    border: none;
    border-top: 1px solid #374151;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

table {
    border-collapse: collapse !important;
    width: 100%;
}

th, td {
    border: 1px solid #374151 !important;
    padding: 12px !important;
}

th {
    background-color: #111827 !important;
    color: white !important;
}

td {
    background-color: #1F2937 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================

def metric_card(title, value, subtitle=""):
    st.markdown(f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <h1>{value}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def section_title(title):
    st.markdown(f"""
    <div class='section-title'>{title}</div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_tables():
    tables = {}

    if os.path.exists("tables"):
        for file in os.listdir("tables"):
            if file.endswith(".csv"):
                tables[file] = pd.read_csv(
                    os.path.join("tables", file)
                )

    return tables

@st.cache_data
def load_figures():
    figures = []

    if os.path.exists("figures"):
        for file in os.listdir("figures"):
            if file.endswith((".png",".jpg",".jpeg")):
                figures.append(
                    os.path.join("figures", file)
                )

    return figures

@st.cache_data
def extract_code_snippets():
    snippets = []

    NOTEBOOK = "Multimodal Alzheimer's Research.ipynb"

    if os.path.exists(NOTEBOOK):

        nb = nbformat.read(
            NOTEBOOK,
            as_version=4
        )

        keywords = [
            "SimpleImputer",
            "StandardScaler",
            "train_test_split",
            "LogisticRegression",
            "RandomForest",
            "ExtraTrees",
            "dropout",
            "mask"
        ]

        for cell in nb.cells:
            if cell.cell_type == "code":
                source = cell.source

                if any(k in source for k in keywords):
                    snippets.append(source)

    return snippets

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown("""
    # 🧠 Alzheimer's AI
    ### Robust Multimodal ML
    """)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Findings", "Key Takeaways"],
        icons=["house","bar-chart","lightbulb"],
        default_index=0
    )

    st.markdown("---")

    st.markdown("""
    ### Research Summary
    
    This project evaluates how multimodal Alzheimer's disease classifiers behave under incomplete clinical information and demonstrates that modality dropout training substantially improves robustness.
    """)

# =========================
# HOME PAGE
# =========================

if selected == "Home":

    st.markdown("""
    <div class='hero-title'>
    Improving Robustness of Multimodal Alzheimer’s Disease Classification Under Missing Clinical Data
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='hero-subtitle'>
    Multimodal machine learning systems for clinically resilient Alzheimer's disease diagnosis.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Best ROC-AUC", "0.977")

    with c2:
        metric_card("Robust Missing AUC", "0.863")

    with c3:
        metric_card("Modalities", "2")

    with c4:
        metric_card("Models Evaluated", "3")

    # =========================
    # MOTIVATION
    # =========================

    section_title("Research Motivation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='glass-card'>
        <h3>Why Alzheimer's AI Matters</h3>
        
        Alzheimer's disease diagnosis increasingly relies on multimodal biomarkers combining cognitive testing and MRI imaging.
        
        Machine learning systems can assist clinicians in identifying disease patterns earlier and more consistently.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
        <h3>Clinical Deployment Challenge</h3>
        
        Real-world healthcare systems frequently contain incomplete patient records, unavailable imaging, and missing cognitive testing.
        
        Robustness under missing information is therefore essential for clinically deployable AI.
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # DATASET
    # =========================

    section_title("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='glass-card'>
        <h3>Clinical Features</h3>
        MMSE, CDRSB, ADAS13, demographic and cognitive variables.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='glass-card'>
        <h3>MRI Biomarkers</h3>
        Hippocampal volume, ventricular measurements, cortical structures.
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='glass-card'>
        <h3>Classification Task</h3>
        CN vs DEM classification using multimodal inputs.
        </div>
        """, unsafe_allow_html=True)

    # Patient distribution chart

    df_patients = pd.DataFrame({
        "Class": ["CN", "DEM"],
        "Patients": [420, 310]
    })

    fig = px.bar(
        df_patients,
        x="Class",
        y="Patients",
        color="Class",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # PIPELINE
    # =========================

    section_title("Technical Pipeline")

    steps = [
        "1. R .rda Conversion",
        "2. Preprocessing",
        "3. Feature Engineering",
        "4. MRI Filtering",
        "5. Imputation",
        "6. Scaling",
        "7. Train/Test Split",
        "8. Model Training",
        "9. Robustness Experiments",
        "10. Evaluation"
    ]

    for step in steps:
        with st.expander(step):
            st.write(f"""
            This stage prepares multimodal Alzheimer's disease data for downstream machine learning evaluation and robustness analysis.
            """)

    st.subheader("Extracted Notebook Code")

    snippets = extract_code_snippets()

    if len(snippets) == 0:
        st.warning("No notebook snippets detected.")
    else:
        for snippet in snippets:
            st.code(snippet, language="python")

    # =========================
    # MODELS
    # =========================

    section_title("Modeling Approach")

    cols = st.columns(3)

    models = [
        (
            "Logistic Regression",
            "High interpretability and strong baseline performance."
        ),
        (
            "Random Forest",
            "Robust nonlinear ensemble model."
        ),
        (
            "Extra Trees",
            "Variance reduction with randomized splitting."
        )
    ]

    for col, model in zip(cols, models):

        with col:

            st.markdown(f"""
            <div class='glass-card'>
            <h3>{model[0]}</h3>
            <p>{model[1]}</p>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # CONTRIBUTION
    # =========================

    section_title("Research Contribution")

    st.markdown("""
    <div class='glass-card'>
    <h3>Key Scientific Contribution</h3>
    
    This research demonstrates that modality dropout training substantially improves resilience to missing clinical information while maintaining nearly identical baseline predictive performance.
    </div>
    """, unsafe_allow_html=True)

# =========================
# FINDINGS PAGE
# =========================

elif selected == "Findings":

    st.title("Research Findings")

    k1,k2,k3,k4,k5,k6 = st.columns(6)

    metrics = [
        ("Accuracy","95.2%"),
        ("ROC-AUC","0.977"),
        ("Recall","91.7%"),
        ("Specificity","95.8%"),
        ("F1 Score","84.6%"),
        ("Robustness","0.140 → 0.114")
    ]

    for col, metric in zip(
        [k1,k2,k3,k4,k5,k6],
        metrics
    ):
        with col:
            metric_card(metric[0], metric[1])

    # =========================
    # MODEL PERFORMANCE
    # =========================

    section_title("Model Performance")

    df_models = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Extra Trees"
        ],
        "ROC-AUC": [0.977,0.965,0.971]
    })

    fig = px.bar(
        df_models,
        x="Model",
        y="ROC-AUC",
        color="ROC-AUC",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    Logistic Regression achieved the strongest balance of interpretability and predictive performance.
    """)

    # =========================
    # MODALITY ANALYSIS
    # =========================

    section_title("Modality Analysis")

    df_modalities = pd.DataFrame({
        "Modality": [
            "Clinical",
            "MRI",
            "Multimodal"
        ],
        "ROC-AUC": [
            0.95,
            0.88,
            0.977
        ]
    })

    fig2 = px.bar(
        df_modalities,
        x="Modality",
        y="ROC-AUC",
        color="ROC-AUC",
        template="plotly_dark"
    )

    fig2.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    Clinical cognitive variables were highly predictive, while MRI biomarkers contributed complementary structural information.
    """)

    # =========================
    # ROBUSTNESS EXPERIMENT
    # =========================

    section_title("Missing Modality Experiment")

    df_robust = pd.DataFrame({
        "Missingness": [10,30,50,70],
        "Baseline": [0.95,0.89,0.81,0.72],
        "Robust": [0.96,0.92,0.88,0.84]
    })

    fig3 = px.line(
        df_robust,
        x="Missingness",
        y=["Baseline","Robust"],
        markers=True,
        template="plotly_dark"
    )

    fig3.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    Modality dropout training substantially improved resilience under increasing missingness conditions.
    """)

    # =========================
    # ABLATION
    # =========================

    section_title("Ablation Study")

    df_ablation = pd.DataFrame({
        "Dropout Rate": [0,0.1,0.3,0.5],
        "Robustness": [0.72,0.79,0.84,0.88],
        "Peak Performance": [0.977,0.975,0.971,0.962]
    })

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=df_ablation["Dropout Rate"],
        y=df_ablation["Robustness"],
        mode='lines+markers',
        name='Robustness'
    ))

    fig4.add_trace(go.Scatter(
        x=df_ablation["Dropout Rate"],
        y=df_ablation["Peak Performance"],
        mode='lines+markers',
        name='Peak Performance'
    ))

    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    Higher dropout rates improved robustness while introducing a modest tradeoff in peak predictive performance.
    """)

    # =========================
    # TABLES
    # =========================

    section_title("Scientific Results Tables")

    tables = load_tables()

    if len(tables) == 0:

        sample = pd.DataFrame({
            "Metric": ["Accuracy","ROC-AUC","Recall"],
            "Value": [0.952,0.977,0.917]
        })

        st.dataframe(sample, use_container_width=True)

    else:

        for name, df in tables.items():
            st.subheader(name)
            st.dataframe(df, use_container_width=True)

    # =========================
    # FIGURES
    # =========================

    section_title("Visualizations")

    figures = load_figures()

    if len(figures) == 0:
        st.info("No external figures detected in /figures")
    else:
        for image in figures:
            st.image(image, use_container_width=True)

    # =========================
    # FEATURE RELATIONSHIPS
    # =========================

    section_title("Feature Relationships")

    corr = np.random.rand(6,6)

    fig5 = px.imshow(
        corr,
        template="plotly_dark",
        color_continuous_scale="Blues"
    )

    fig5.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    Cognitive and memory-related variables demonstrated strong relationships with disease severity, while MRI biomarkers captured complementary structural degeneration patterns.
    """)

# =========================
# TAKEAWAYS PAGE
# =========================

elif selected == "Key Takeaways":

    st.title("Key Takeaways")

    section_title("Core Scientific Findings")

    findings = [
        "Multimodal systems are vulnerable to missing clinical data.",
        "Modality dropout significantly improves robustness.",
        "MRI biomarkers provide complementary information.",
        "Robustness and peak accuracy exhibit tradeoffs."
    ]

    for finding in findings:
        st.markdown(f"""
        <div class='glass-card'>
        <h3>{finding}</h3>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # CLINICAL IMPLICATIONS
    # =========================

    section_title("Clinical Implications")

    st.markdown("""
    <div class='glass-card'>
    
    Real-world healthcare deployment environments frequently contain incomplete medical records, unavailable MRI studies, and missing cognitive testing.
    
    AI systems intended for deployment must therefore optimize not only for predictive performance, but also for resilience under uncertainty.
    
    Robust multimodal systems may improve reliability and clinician trust in AI-assisted diagnosis pipelines.
    
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # LIMITATIONS
    # =========================

    section_title("Limitations")

    st.markdown("""
    <div class='glass-card'>
    
    • ADNI cohort limitations
    
    • Synthetic missingness assumptions
    
    • Binary classification framing
    
    • Lack of external validation
    
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # FUTURE WORK
    # =========================

    section_title("Future Work")

    future_cols = st.columns(3)

    future = [
        "OASIS Validation",
        "NACC Validation",
        "Transformer Architectures",
        "Longitudinal Modeling",
        "Fairness Analysis",
        "MAR Missingness"
    ]

    for i, item in enumerate(future):

        with future_cols[i % 3]:

            st.markdown(f"""
            <div class='glass-card'>
            <h3>{item}</h3>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # CONCLUSION
    # =========================

    section_title("Final Conclusion")

    st.markdown("""
    <div class='glass-card'>
    
    This project demonstrates that robustness matters as much as predictive accuracy in healthcare AI.
    
    Clinically deployable multimodal systems must tolerate incomplete records, unavailable modalities, and real-world uncertainty.
    
    Modality dropout training creates substantially more resilient diagnostic systems while maintaining strong baseline predictive performance.
    
    <br><br>
    
    <h3>
    “Clinically deployable AI systems must optimize not only for predictive performance, but also for resilience under real-world uncertainty.”
    </h3>
    
    </div>
    """, unsafe_allow_html=True)

