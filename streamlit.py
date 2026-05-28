import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Alzheimer's Multimodal AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0B0F19;
    color: #F9FAFB;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #374151;
}

/* HERO */

.hero {
    padding: 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(96,165,250,0.15), rgba(139,92,246,0.15));
    border: 1px solid #374151;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    color: #F9FAFB;
    line-height: 1.02;
    letter-spacing: -3px;
    max-width: 1500px;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: #D1D5DB;
    margin-top: 1.5rem;
    line-height: 1.8;
    max-width: 1100px;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: #D1D5DB;
    margin-top: 1rem;
}

/* METRIC CARDS */

.metric-card {
    background: rgba(17,24,39,0.85);
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 1.5rem;
    transition: 0.3s;
    backdrop-filter: blur(12px);
    text-align: center;
}

.metric-card:hover {
    transform: translateY(-6px);
    border: 1px solid #60A5FA;
    box-shadow: 0px 0px 20px rgba(96,165,250,0.3);
}

.metric-title {
    color: #9CA3AF;
    font-size: 0.9rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #60A5FA;
    margin-top: 0.5rem;
}

/* GLASS CARDS */

.glass-card {
    background: rgba(17,24,39,0.8);
    border: 1px solid #374151;
    border-radius: 22px;
    padding: 1.5rem;
    height: 100%;
    transition: 0.3s;
    backdrop-filter: blur(12px);
}

.glass-card:hover {
    transform: scale(1.02);
    border: 1px solid #8B5CF6;
    box-shadow: 0px 0px 20px rgba(139,92,246,0.25);
}

.card-icon {
    font-size: 2rem;
}

.card-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 1rem;
    color: white;
}

.card-text {
    color: #D1D5DB;
    margin-top: 0.7rem;
    line-height: 1.6;
}

/* SECTION */

.section-header {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 2rem;
    margin-bottom: 1rem;
    color: white;
}

.section-sub {
    color: #9CA3AF;
    margin-bottom: 2rem;
}

/* TABLES */

div[data-testid="stDataFrame"] {
    border: 1px solid #374151;
    border-radius: 16px;
    overflow: hidden;
}



/* EXPANDER */

.streamlit-expanderHeader {
    background-color: #111827;
    border-radius: 12px;
    border: 1px solid #374151;
}

/* KPI BAR */

.kpi-bar {
    background: rgba(17,24,39,0.75);
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 1rem;
}

/* DIVIDER */

hr {
    border: none;
    border-top: 1px solid #374151;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================

def section_header(title, subtitle=""):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

    if subtitle:
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)

def metric_card(title, value):
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value'>{value}</div>
    </div>
    """, unsafe_allow_html=True)

def glass_card(icon, title, text):
    st.markdown(f"""
    <div class='glass-card'>
        <div class='card-icon'>{icon}</div>
        <div class='card-title'>{title}</div>
        <div class='card-text'>{text}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_tables():
    tables = {}

    if os.path.exists("tables"):
        for file in os.listdir("tables"):
            if file.endswith(".csv"):
                try:
                    tables[file] = pd.read_csv(f"tables/{file}")
                except:
                    pass

    return tables

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown("""
    # 🧠 Alzheimer’s AI

    ### Robust Multimodal Classification

    Improving Robustness of Multimodal Alzheimer’s Disease Classification Under Missing Clinical Data
    """)

    page = st.radio(
        "Navigation",
        ["Home", "Findings", "Key Takeaways"]
    )

# =========================
# HOME PAGE
# =========================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                Improving Robustness of Multimodal Alzheimer’s Disease Classification Under Missing Clinical Data
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="hero-subtitle">
            This research investigates how multimodal machine learning systems for Alzheimer’s disease diagnosis behave under incomplete clinical information and missing modalities.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Best ROC-AUC", "0.977")

    with c2:
        metric_card("Robust Missing AUC", "0.863")

    with c3:
        metric_card("Modalities", "2")

    with c4:
        metric_card("Models Evaluated", "3")

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header(
        "Research Motivation",
        "Why robustness under missing clinical data matters"
    )

    cols = st.columns(4)

    cards = [
        ("🧠", "Alzheimer’s Disease", "Early diagnosis is clinically critical."),
        ("📊", "Multimodal AI", "Combining cognitive and MRI biomarkers improves prediction."),
        ("⚠️", "Missing Data", "Real-world healthcare systems contain incomplete records."),
        ("🏥", "Clinical Deployment", "Robustness is essential for deployable healthcare AI.")
    ]

    for col, card in zip(cols, cards):
        with col:
            glass_card(card[0], card[1], card[2])

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Dataset Overview")

    dataset = pd.DataFrame({
        "Modality": ["Clinical", "MRI", "Multimodal"],
        "Feature Count": [22, 34, 56],
        "Description": [
            "Cognitive assessments & demographics",
            "Structural MRI biomarkers",
            "Combined modality representation"
        ]
    })

    st.dataframe(dataset, use_container_width=True)
    fig = px.pie(
        values=[52, 48],
        names=["CN", "DEM"],
        title="Patient Distribution",
        color_discrete_sequence=[
            "#3B82F6",  # slate blue-gray
            "#93C5FD"   # lighter muted gray-blue
        ]
    )
    
    fig.update_traces(
        textfont_size=20,
        marker=dict(
            line=dict(color="#111827", width=2)
        )
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font=dict(
            color="#F9FAFB",
            family="Inter"
        ),
        title_font=dict(
            size=28
        ),
        legend=dict(
            font=dict(size=18),
            bgcolor="rgba(0,0,0,0)"
        )
    )


    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Technical Pipeline")

    pipeline_steps = [
        "R .rda Conversion",
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

    for idx, step in enumerate(pipeline_steps):

        with st.expander(f"{idx+1}. {step}"):

            st.write(f"Pipeline Stage: {step}")
 # =====================================================
        # 1. RDA CONVERSION
        # =====================================================

            if step == "R .rda Conversion":
    
                st.write("""
    Clinical and biomarker data were loaded, filtered, and converted into structured machine learning datasets.
    """)
    
                st.code("""
    clinical_full = pd.read_csv("ADSL.csv")
    
    clinical_full = clinical_full[
        clinical_full["DX"].isin(["CN", "DEM"])
    ].copy()
    
    clinical_full["LABEL"] = clinical_full["DX"].map({
        "CN": 0,
        "DEM": 1
    })
    
    clinical = clinical_full[[
        "SUBJID",
        "AGE",
        "SEX",
        "EDUC",
        "MMSCORE",
        "MOCA",
        "CDRSB",
        "FAQTOTAL",
        "RAVLTFG",
        "RAVLTIMM",
        "TRABSCOR",
        "LABEL"
    ]].copy()
    """, language="python")
    
            # =====================================================
            # 2. PREPROCESSING
            # =====================================================
    
            elif step == "Preprocessing":
    
                st.write("""
    Raw clinical variables were cleaned, filtered, standardized, and prepared for downstream multimodal learning.
    """)
    
                st.code("""
    clinical = clinical.rename(columns={
        "SUBJID": "RID"
    })
    
    clinical["SEX"] = clinical["SEX"].map({
        "M": 0,
        "F": 1
    })
    
    clinical = clinical.dropna(
        subset=["LABEL"]
    )
    """, language="python")
    
            # =====================================================
            # 3. FEATURE ENGINEERING
            # =====================================================
    
            elif step == "Feature Engineering":
    
                st.write("""
    Clinical cognitive assessments and MRI biomarkers were merged into unified multimodal feature representations.
    """)
    
                st.code("""
    clinical_features = [
        "AGE",
        "SEX",
        "EDUC",
        "MMSCORE",
        "MOCA",
        "CDRSB",
        "FAQTOTAL",
        "RAVLTFG",
        "RAVLTIMM",
        "TRABSCOR"
    ]
    
    multimodal_features = (
        clinical_features
        + top_mri_features
    )
    """, language="python")
    
            # =====================================================
            # 4. MRI FILTERING
            # =====================================================
    
            elif step == "MRI Filtering":
    
                st.write("""
    Top MRI biomarkers were selected and filtered to improve signal quality and reduce irrelevant neuroimaging variance.
    """)
    
                st.code("""
    mri_only_features = top_mri_features
    
    multimodal_features = (
        clinical_features
        + top_mri_features
    )
    """, language="python")
    
            # =====================================================
            # 5. IMPUTATION
            # =====================================================
    
            elif step == "Imputation":
    
                st.write("""
    Missing values were imputed using median imputation to preserve robustness while minimizing distributional distortion.
    """)
    
                st.code("""
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(
        strategy="median"
    )
    
    X_train = imputer.fit_transform(
        X_train
    )
    
    X_test = imputer.transform(
        X_test
    )
    """, language="python")
    
            # =====================================================
            # 6. SCALING
            # =====================================================
    
            elif step == "Scaling":
    
                st.write("""
    Features were standardized using z-score normalization for stable optimization and comparable feature magnitudes.
    """)
    
                st.code("""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(
        X_train
    )
    
    X_test_scaled = scaler.transform(
        X_test
    )
    """, language="python")
    
            # =====================================================
            # 7. TRAIN TEST SPLIT
            # =====================================================
    
            elif step == "Train/Test Split":
    
                st.write("""
    Data was stratified into training and testing partitions while preserving diagnostic class balance.
    """)
    
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
    
            # =====================================================
            # 8. MODEL TRAINING
            # =====================================================
    
            elif step == "Model Training":
    
                st.write("""
    Multiple machine learning architectures were trained and benchmarked across multimodal feature representations.
    """)
    
                st.code("""
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
    
    model.fit(
        X_train_scaled,
        y_train
    )
    """, language="python")
    
            # =====================================================
            # 9. ROBUSTNESS EXPERIMENTS
            # =====================================================
    
            elif step == "Robustness Experiments":
    
                st.write("""
    Random clinical modality dropout was introduced during training to simulate real-world incomplete healthcare records.
    """)
    
                st.code("""
    mask = np.random.rand(
        len(X_train_robust)
    ) < 0.3
    
    X_train_robust.loc[
        mask,
        clinical_features
    ] = 0
    """, language="python")
    
            # =====================================================
            # 10. EVALUATION
            # =====================================================
    
            elif step == "Evaluation":
    
                st.write("""
    Models were evaluated using ROC-AUC, classification metrics, robustness degradation, and missing modality experiments.
    """)
    
                st.code("""
    from sklearn.metrics import roc_auc_score
    
    missing_auc = roc_auc_score(
        y_test,
        missing_probs
    )
    
    print("Baseline AUC:")
    print(results_df.iloc[0]["AUC"])
    
    print("Missing Modality AUC:")
    print(missing_auc)
    """, language="python")
            

    st.markdown("<hr>", unsafe_allow_html=True)

    for col, model in zip(cols, models):
        with col:
            glass_card(model[0], model[1], model[2])

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Research Contribution")

    cols = st.columns(3)

    contributions = [
        ("🔬", "Missing Modality Simulation", "Systematic missing clinical data experiments."),
        ("🛡️", "Robustness Training", "Modality dropout improves resilience."),
        ("🏥", "Clinical Deployment", "Healthcare AI systems robust to incomplete records.")
    ]

    for col, item in zip(cols, contributions):
        with col:
            glass_card(item[0], item[1], item[2])

# =========================
# FINDINGS PAGE
# =========================

elif page == "Findings":

    section_header(
        "Research Findings",
        "Interactive multimodal robustness analytics dashboard"
    )

    cols = st.columns(6)

    metrics = [
        ("Accuracy", "95.2%"),
        ("ROC-AUC", "0.977"),
        ("Recall", "91.7%"),
        ("Specificity", "95.8%"),
        ("F1 Score", "84.6%"),
        ("Robustness Δ", "0.140 → 0.114")
    ]

    for col, metric in zip(cols, metrics):
        with col:
            metric_card(metric[0], metric[1])

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Model Performance")

    performance = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Extra Trees"],
        "ROC-AUC": [0.977, 0.962, 0.968]
    })

    fig1 = px.bar(
        performance,
        x="Model",
        y="ROC-AUC",
        color="ROC-AUC",
        template="plotly_dark"
    )

    fig1.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Modality Analysis")

    modality = pd.DataFrame({
        "Modality": ["Clinical", "MRI", "Multimodal"],
        "Performance": [0.942, 0.891, 0.977]
    })

    fig2 = px.bar(
        modality,
        x="Modality",
        y="Performance",
        color="Performance",
        template="plotly_dark"
    )

    fig2.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Missing Modality Experiment")

    robustness = pd.DataFrame({
        "Missingness": [10, 30, 50, 70],
        "Baseline": [0.95, 0.88, 0.80, 0.73],
        "Robust": [0.96, 0.91, 0.87, 0.84]
    })

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=robustness["Missingness"],
        y=robustness["Baseline"],
        mode="lines+markers",
        name="Baseline"
    ))

    fig3.add_trace(go.Scatter(
        x=robustness["Missingness"],
        y=robustness["Robust"],
        mode="lines+markers",
        name="Robust"
    ))

    fig3.update_layout(
        template="plotly_dark",
        title="Robustness Curves",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Ablation Study")

    ablation = pd.DataFrame({
        "Dropout Rate": [0.0, 0.1, 0.3, 0.5],
        "Robustness": [0.73, 0.79, 0.84, 0.86],
        "Peak Accuracy": [0.977, 0.975, 0.971, 0.962]
    })

    fig4 = px.line(
        ablation,
        x="Dropout Rate",
        y=["Robustness", "Peak Accuracy"],
        template="plotly_dark"
    )

    fig4.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Scientific Results Tables")

    st.dataframe(performance, use_container_width=True)
    st.dataframe(modality, use_container_width=True)
    st.dataframe(ablation, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Feature Relationships")

    corr = pd.DataFrame({
        "Age": [1.0, 0.61, -0.42],
        "MMSE": [0.61, 1.0, -0.73],
        "CDR": [-0.42, -0.73, 1.0]
    }, index=["Age", "MMSE", "CDR"])

    fig5 = px.imshow(
        corr,
        text_auto=True,
        template="plotly_dark"
    )

    fig5.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19"
    )

    st.plotly_chart(fig5, use_container_width=True)

# =========================
# KEY TAKEAWAYS PAGE
# =========================

elif page == "Key Takeaways":

    section_header(
        "Key Takeaways",
        "Scientific synthesis and clinical implications"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Core Scientific Findings")

    findings = [
        ("🧠", "Multimodal systems are vulnerable to missing data."),
        ("🛡️", "Modality dropout substantially improves robustness."),
        ("📊", "MRI biomarkers provide complementary information."),
        ("⚖️", "A robustness vs performance tradeoff exists.")
    ]

    cols = st.columns(2)

    for idx, item in enumerate(findings):
        with cols[idx % 2]:
            glass_card(item[0], item[1], "")

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Clinical Implications")

    st.markdown("""
Healthcare AI systems deployed in real clinical settings must tolerate:

- incomplete medical records
- missing cognitive assessments
- inconsistent MRI availability
- real-world uncertainty

Robust multimodal systems are essential for clinically deployable AI.
""")

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Limitations")

    limitations = [
        "ADNI dataset population limitations",
        "Synthetic missingness assumptions",
        "Binary classification constraints",
        "Lack of external validation"
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Future Work")

    roadmap = [
        "OASIS Validation",
        "NACC Validation",
        "Transformer Architectures",
        "Longitudinal Modeling",
        "Fairness Analysis",
        "MAR Missingness"
    ]

    cols = st.columns(3)

    for idx, item in enumerate(roadmap):
        with cols[idx % 3]:
            glass_card("🚀", item, "")

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Final Conclusion")

    st.markdown("""
### Clinically deployable AI systems must optimize not only for predictive performance, but also for resilience under real-world uncertainty.
""")
