import os
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
    font-family: sans-serif;
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
    transition: 0.3s ease;
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
    margin-bottom: 1rem;
}

.section-title {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 3rem;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #9CA3AF;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================

def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <h1>{value}</h1>
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
    Robust multimodal AI systems for Alzheimer's disease diagnosis under missing clinical information.
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
    Multimodal machine learning systems for clinically resilient Alzheimer's diagnosis.
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
        metric_card("Models Tested", "3")

    section_title("Research Motivation")

    st.markdown("""
    <div class='glass-card'>
    
    Alzheimer's disease diagnosis increasingly relies on multimodal AI systems combining cognitive testing and MRI biomarkers.
    
    Real-world healthcare environments frequently contain incomplete records and missing clinical data.
    
    This project evaluates robustness under missing information and demonstrates that modality dropout substantially improves resilience.
    
    </div>
    """, unsafe_allow_html=True)

    section_title("Dataset Overview")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='glass-card'>
        <h3>Clinical Variables</h3>
        MMSE, ADAS13, CDRSB, demographics.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='glass-card'>
        <h3>MRI Biomarkers</h3>
        Structural neuroimaging measurements.
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='glass-card'>
        <h3>Classification</h3>
        CN vs DEM prediction task.
        </div>
        """, unsafe_allow_html=True)

    section_title("Technical Pipeline")

    steps = [
        "R .rda conversion",
        "Preprocessing",
        "Feature Engineering",
        "MRI Filtering",
        "Imputation",
        "Scaling",
        "Train/Test Split",
        "Model Training",
        "Robustness Experiments",
        "Evaluation"
    ]

    for step in steps:
        with st.expander(step):
            st.write(f"Pipeline stage: {step}")

    st.subheader("Core Modeling Code")

    st.code("""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

imputer = SimpleImputer(strategy='mean')

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)
""", language="python")

    section_title("Modeling Approach")

    cols = st.columns(3)

    models = [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees"
    ]

    for i, model in enumerate(models):

        with cols[i]:

            st.markdown(f"""
            <div class='glass-card'>
            <h3>{model}</h3>
            </div>
            """, unsafe_allow_html=True)

    section_title("Research Contribution")

    st.markdown("""
    <div class='glass-card'>
    
    Modality dropout training substantially improves robustness under missing clinical information while maintaining nearly identical baseline predictive performance.
    
    </div>
    """, unsafe_allow_html=True)

# =========================
# FINDINGS PAGE
# =========================

elif selected == "Findings":

    st.title("Research Findings")

    cols = st.columns(6)

    metrics = [
        ("Accuracy","95.2%"),
        ("ROC-AUC","0.977"),
        ("Recall","91.7%"),
        ("Specificity","95.8%"),
        ("F1 Score","84.6%"),
        ("Robustness","0.140 → 0.114")
    ]

    for col, metric in zip(cols, metrics):

        with col:
            metric_card(metric[0], metric[1])

    section_title("Model Performance")

    df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Extra Trees"
        ],
        "ROC-AUC": [
            0.977,
            0.965,
            0.971
        ]
    })

    fig = px.bar(
        df,
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

    section_title("Missingness Robustness")

    robust_df = pd.DataFrame({
        "Missingness": [10,30,50,70],
        "Baseline": [0.95,0.89,0.81,0.72],
        "Robust": [0.96,0.92,0.88,0.84]
    })

    fig2 = px.line(
        robust_df,
        x="Missingness",
        y=["Baseline","Robust"],
        markers=True,
        template="plotly_dark"
    )

    fig2.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig2, use_container_width=True)

    section_title("Scientific Tables")

    tables = load_tables()

    if len(tables) == 0:

        sample = pd.DataFrame({
            "Metric": ["Accuracy","ROC-AUC","Recall"],
            "Value": [0.952,0.977,0.917]
        })

        st.dataframe(sample)

    else:

        for name, df in tables.items():

            st.subheader(name)

            st.dataframe(
                df,
                use_container_width=True
            )

    section_title("Visualizations")

    figures = load_figures()

    if len(figures) == 0:
        st.info("No figures detected.")
    else:
        for image in figures:
            st.image(image, use_container_width=True)

# =========================
# TAKEAWAYS PAGE
# =========================

elif selected == "Key Takeaways":

    st.title("Key Takeaways")

    findings = [
        "Multimodal systems are vulnerable to missing data.",
        "Modality dropout improves robustness.",
        "MRI biomarkers provide complementary information.",
        "Robustness vs accuracy tradeoffs exist."
    ]

    for finding in findings:

        st.markdown(f"""
        <div class='glass-card'>
        <h3>{finding}</h3>
        </div>
        """, unsafe_allow_html=True)

    section_title("Clinical Implications")

    st.markdown("""
    <div class='glass-card'>
    
    Clinically deployable AI systems must remain stable despite incomplete patient records and missing modalities.
    
    </div>
    """, unsafe_allow_html=True)

    section_title("Limitations")

    st.markdown("""
    <div class='glass-card'>
    
    • ADNI limitations
    
    • Synthetic missingness
    
    • Binary classification framing
    
    • No external validation
    
    </div>
    """, unsafe_allow_html=True)

    section_title("Future Work")

    future = [
        "OASIS Validation",
        "NACC Validation",
        "Transformer Architectures",
        "Longitudinal Modeling",
        "Fairness Analysis",
        "MAR Missingness"
    ]

    cols = st.columns(3)

    for i, item in enumerate(future):

        with cols[i % 3]:

            st.markdown(f"""
            <div class='glass-card'>
            <h3>{item}</h3>
            </div>
            """, unsafe_allow_html=True)

    section_title("Final Conclusion")

    st.markdown("""
    <div class='glass-card'>
    
    Clinically deployable AI systems must optimize not only for predictive performance, but also for resilience under real-world uncertainty.
    
    </div>
    """, unsafe_allow_html=True)

