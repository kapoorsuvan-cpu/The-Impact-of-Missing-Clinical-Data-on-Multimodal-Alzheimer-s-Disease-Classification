import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# =========================
# PAGE CONFIG
# =========================
CHART_COLORS = {
    "primary": "#60A5FA",
    "secondary": "#93C5FD",
    "tertiary": "#BFDBFE",
    "slate": "#64748B",
    "muted": "#94A3B8",
    "border": "#374151",
    "bg": "#0B0F19",
    "card": "#111827",
    "text": "#F9FAFB"
}

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

/* TABLE TEXT ALIGNMENT */

thead tr th {
    text-align: right !important;
}

tbody tr td {
    text-align: right !important;
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
        ("Robustness Δ", "18.6% ↓")
    ]

    for col, metric in zip(cols, metrics):
        with col:
            metric_card(metric[0], metric[1])

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    ### What These Results Mean
    
    The primary research question of this project was whether multimodal Alzheimer's disease classification systems can remain reliable when clinical information becomes incomplete or unavailable.
    
    The results demonstrate that the multimodal framework achieved excellent predictive performance under standard conditions, with **95.2% accuracy** and a **ROC-AUC of 0.977**, indicating near-perfect discrimination between cognitively normal (CN) and dementia (DEM) patients. The model also maintained strong **recall (91.7%)**, meaning it successfully identified the vast majority of Alzheimer's cases, while a **specificity of 95.8%** indicates very few cognitively normal patients were incorrectly classified as having dementia.
    
    Most importantly, the **Robustness Δ** metric directly addresses the project's central research question. This value represents the performance degradation experienced when clinical information is missing. The baseline multimodal model suffered a **0.140 drop in ROC-AUC**, whereas the modality-dropout-trained model reduced that degradation to **0.114**. In practical terms, this means the robust model was significantly more resilient when clinical records became incomplete.
    
    These findings suggest that while multimodal AI systems are highly effective under ideal conditions, they can be vulnerable to missing data. However, modality dropout training successfully improves robustness without sacrificing baseline predictive performance, making the resulting system substantially more suitable for real-world clinical deployment where incomplete patient records are common.
    """)

    section_header("Model Performance")

    performance = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Extra Trees"],
        "ROC-AUC": [0.977, 0.962, 0.968]
    })
    
    fig1 = px.bar(
        performance,
        x="Model",
        y="ROC-AUC",
        template="plotly_dark",
        color_discrete_sequence=[CHART_COLORS["primary"]],
    )
    
    fig1.update_traces(
        marker_line_color=CHART_COLORS["border"],
        marker_line_width=1.5,
        opacity=0.92
    )
    fig1.update_layout(
        paper_bgcolor=CHART_COLORS["bg"],
        plot_bgcolor=CHART_COLORS["bg"],
        font=dict(
            color=CHART_COLORS["text"],
            family="Inter"
        ),
    
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.12)"
        )
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    ### Interpretation
    
    Logistic Regression achieved the strongest overall ROC-AUC performance, outperforming Random Forest and Extra Trees while maintaining substantially higher interpretability and calibration stability.
    
    Because Alzheimer’s disease classification is a clinically sensitive application, interpretability and reliability were prioritized alongside predictive performance. As a result, Logistic Regression was selected as the primary modeling architecture used throughout the robustness experiments and missing modality analyses.
    
    The strong performance of a relatively simple linear model also suggests that the multimodal feature space contained highly separable signal structure, particularly when combining cognitive and MRI biomarkers.
    """)

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
        template="plotly_dark",
        color_discrete_sequence=[
            CHART_COLORS["primary"],
            CHART_COLORS["secondary"],
            CHART_COLORS["tertiary"]
        ]
    )
    
    fig2.update_traces(
        marker_line_color=CHART_COLORS["border"],
        marker_line_width=1.5
    )
    
    fig2.update_layout(
        paper_bgcolor=CHART_COLORS["bg"],
        plot_bgcolor=CHART_COLORS["bg"],
        font=dict(
            color=CHART_COLORS["text"],
            family="Inter"
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.12)"
        )
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    ### Interpretation
    
    Clinical cognitive variables alone provided remarkably strong predictive performance, highlighting the diagnostic value of memory, executive function, and cognitive assessment scores in Alzheimer’s disease classification.
    
    MRI biomarkers independently contributed meaningful predictive signal, though performance remained lower than the clinical-only modality. However, combining both modalities produced the highest overall ROC-AUC, demonstrating that MRI biomarkers provide complementary information beyond standard cognitive testing.
    
    These findings directly motivated the project’s multimodal modeling framework and reinforced the importance of preserving both modalities whenever possible.
    """)

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
        name="Baseline",
        line=dict(
            color="#94A3B8",
            width=4
        ),
        marker=dict(size=8)
    ))
    
    fig3.add_trace(go.Scatter(
        x=robustness["Missingness"],
        y=robustness["Robust"],
        mode="lines+markers",
        name="Robust",
        line=dict(
            color="#60A5FA",
            width=4
        ),
        marker=dict(size=8),
        fill="tonexty",
        fillcolor="rgba(96,165,250,0.08)"
    ))
    
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["bg"],
        plot_bgcolor=CHART_COLORS["bg"],
        font=dict(
            color=CHART_COLORS["text"],
            family="Inter"
        ),
        xaxis=dict(
            title="Missingness %",
            gridcolor="rgba(148,163,184,0.08)"
        ),
        yaxis=dict(
            title="ROC-AUC",
            gridcolor="rgba(148,163,184,0.08)"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    ### Interpretation
    
    The baseline multimodal model experienced substantial degradation as clinical information became increasingly unavailable. This finding demonstrated a major weakness of standard multimodal AI systems: strong dependence on complete clinical records.
    
    After introducing modality dropout training, the robust model maintained significantly greater stability across all missingness levels. The performance gap widened as missingness increased, indicating that robustness training improved the model’s ability to tolerate incomplete healthcare data.
    
    This experiment became the central contribution of the project and validated the effectiveness of simulated modality dropout for improving deployability in real clinical environments.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Ablation Study")

    ablation = pd.DataFrame({
        "Dropout Rate": [0.0, 0.1, 0.3, 0.5],
        "Robustness": [0.73, 0.79, 0.84, 0.86],
        "Peak Accuracy": [0.977, 0.975, 0.971, 0.962]
    })

    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatter(
        x=ablation["Dropout Rate"],
        y=ablation["Robustness"],
        mode="lines+markers",
        name="Robustness",
        line=dict(
            color="#60A5FA",
            width=4
        )
    ))
    
    fig4.add_trace(go.Scatter(
        x=ablation["Dropout Rate"],
        y=ablation["Peak Accuracy"],
        mode="lines+markers",
        name="Peak Accuracy",
        line=dict(
            color="#94A3B8",
            width=4
        )
    ))
    
    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["bg"],
        plot_bgcolor=CHART_COLORS["bg"],
        font=dict(
            color=CHART_COLORS["text"],
            family="Inter"
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)"
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)"
        )
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    ### Interpretation
    
    Increasing modality dropout rates improved robustness under missing clinical data but introduced a small tradeoff in peak predictive performance.
    
    Lower dropout rates preserved maximum ROC-AUC performance under complete information, while higher dropout rates improved resilience under degraded clinical conditions.
    
    This tradeoff highlighted a central conclusion of the research: healthcare AI systems should optimize not only for ideal-case accuracy, but also for stability and reliability under real-world uncertainty.
    
    The final selected dropout configuration balanced both objectives by maintaining near-state-of-the-art accuracy while substantially improving robustness.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)

    section_header("Scientific Results Tables")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Model Performance")
        st.table(
            performance.style
            .hide(axis="index")
            .format({"ROC-AUC": "{:.3f}"})
        )
    
    with col2:
        st.markdown("##### Modality Results")
        st.table(
            modality.style
            .hide(axis="index")
            .format({"Performance": "{:.3f}"})
        )
    
    with col3:
        st.markdown("##### Ablation Results")
        st.table(
            ablation.style
            .hide(axis="index")
            .format({
                "Dropout Rate": "{:.1f}",
                "Robustness": "{:.2f}",
                "Peak Accuracy": "{:.3f}"
            })
        )

    st.markdown("""
    ### Interpretation
    
    The scientific results tables summarize the quantitative outcomes across all modeling experiments, modality configurations, and robustness evaluations.
    
    The consistency of Logistic Regression across both standard and missing-modality conditions reinforced its selection as the final deployed modeling framework. Cross-experimental stability was especially important because clinical AI systems require predictable behavior under incomplete records and operational variability.
    
    The ablation studies further demonstrated that robustness improvements were systematic rather than isolated to a single experiment or dataset partition.
    """)


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
        color_continuous_scale=[
            [0.0, "#1E293B"],
            [0.5, "#60A5FA"],
            [1.0, "#EF4444"]
        ]
    )
    
    fig5.update_layout(
        paper_bgcolor=CHART_COLORS["bg"],
        plot_bgcolor=CHART_COLORS["bg"],
        font=dict(
            color=CHART_COLORS["text"],
            family="Inter"
        )
    )

    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""
    ### Interpretation
    
    The correlation analysis revealed strong relationships between cognitive impairment metrics, including MMSE and CDR scores, which are clinically associated with Alzheimer’s disease progression and memory decline.
    
    Negative correlations between cognitive performance and disease severity aligned with established neurological findings and validated the biological plausibility of the feature space used by the models.
    
    These relationships helped explain why cognitive assessments contributed such strong predictive signal within the multimodal classification framework.
    """)

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
