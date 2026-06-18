import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Parkinson's Voice Biomarker Dashboard",
    page_icon="🎙️",
    layout="wide"
)


# --------------------------------------------------
# Theme / Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: #e5e7eb;
    }

    /* Main content padding */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Main text */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #e5e7eb;
    }

    /* Page title */
    .page-title {
        font-size: 2.8rem;
        font-weight: 850;
        color: #f8fafc;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem;
    }

    .page-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        max-width: 950px;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 1.25rem 1.35rem;
        border-radius: 20px;
        border: 1px solid #334155;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        min-height: 112px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2.15rem;
        font-weight: 850;
        letter-spacing: -0.03em;
    }

    .section-card {
        background: rgba(15, 23, 42, 0.92);
        padding: 1.45rem 1.6rem;
        border-radius: 20px;
        border: 1px solid #334155;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.25rem;
    }

    .section-card h3 {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }

    .section-card p, .section-card li {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.65;
    }

    .section-card b {
        color: #67e8f9;
    }

    .warning-box {
        background: rgba(234, 88, 12, 0.13);
        border: 1px solid rgba(251, 146, 60, 0.45);
        border-left: 6px solid #fb923c;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        color: #fed7aa;
        margin-bottom: 1.25rem;
    }

    .warning-box b {
        color: #ffedd5;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Streamlit built-in elements */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid #334155;
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #334155;
    }

    /* Select boxes / multiselects */
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #e5e7eb !important;
    }

    div[data-baseweb="select"] span {
        color: #e5e7eb !important;
    }

    /* Expander */
    details {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 16px !important;
    }

    summary {
        color: #e5e7eb !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #67e8f9 !important;
        border-bottom-color: #67e8f9 !important;
    }

    /* Horizontal rule */
    hr {
        border-color: #334155;
    }

    /* Remove white blocks from markdown/html cards */
    .element-container {
        color: #e5e7eb;
    }

    /* Make Plotly chart containers feel integrated */
    div[data-testid="stPlotlyChart"] {
        background: rgba(15, 23, 42, 0.25);
        border-radius: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = "data/processed/parkinsons_cleaned.csv"
MODELING_DIR = "reports/modeling"
MODEL_PATH = "models/parkinsons_voice_model.pkl"


# --------------------------------------------------
# Plotly theme helper
# --------------------------------------------------

PLOT_TEMPLATE = "plotly_dark"

COLOR_MAP = {
    "Healthy": "#38bdf8",
    "Parkinson's": "#818cf8"
}

SEQUENTIAL_BLUE = px.colors.sequential.Blues
DIVERGING = px.colors.diverging.RdBu


def style_plot(fig):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="rgba(15, 23, 42, 0.72)",
        font=dict(color="#e5e7eb"),
        title=dict(
            font=dict(size=22, color="#f8fafc"),
            x=0.02,
            xanchor="left"
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e5e7eb")
        ),
        margin=dict(l=30, r=30, t=70, b=45)
    )

    fig.update_xaxes(
        gridcolor="#334155",
        zerolinecolor="#475569",
        linecolor="#475569",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#e5e7eb")
    )

    fig.update_yaxes(
        gridcolor="#334155",
        zerolinecolor="#475569",
        linecolor="#475569",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#e5e7eb")
    )

    return fig


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    status_labels = {
        0: "Healthy",
        1: "Parkinson's"
    }

    df["diagnosis_label"] = df["status"].map(status_labels)

    return df


@st.cache_data
def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model_if_exists():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


df = load_data()

model_comparison = load_csv_if_exists(
    os.path.join(MODELING_DIR, "train_test_model_comparison.csv")
)

cv_results = load_csv_if_exists(
    os.path.join(MODELING_DIR, "grouped_cv_model_comparison.csv")
)

feature_group_results = load_csv_if_exists(
    os.path.join(MODELING_DIR, "feature_group_comparison.csv")
)

logistic_coefficients = load_csv_if_exists(
    os.path.join(MODELING_DIR, "logistic_regression_coefficients.csv")
)

rf_importance = load_csv_if_exists(
    os.path.join(MODELING_DIR, "random_forest_feature_importance.csv")
)

saved_model = load_model_if_exists()


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

frequency_features = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)"
]

jitter_features = [
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP"
]

shimmer_features = [
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA"
]

noise_features = [
    "NHR",
    "HNR"
]

nonlinear_features = [
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]

all_feature_columns = (
    frequency_features +
    jitter_features +
    shimmer_features +
    noise_features +
    nonlinear_features
)

feature_group_map = {
    "Frequency": frequency_features,
    "Jitter": jitter_features,
    "Shimmer": shimmer_features,
    "Noise": noise_features,
    "Nonlinear": nonlinear_features,
    "All Features": all_feature_columns
}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def section_header(title, subtitle=None):
    st.markdown(
        f"""
        <div class="page-title">{title}</div>
        """,
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f"""
            <div class="page-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True
        )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_patient_level_df(data):
    patient_df = (
        data.groupby(["patient_id", "status"], as_index=False)[all_feature_columns]
        .median()
    )

    patient_df["diagnosis_label"] = patient_df["status"].map({
        0: "Healthy",
        1: "Parkinson's"
    })

    return patient_df


patient_df = create_patient_level_df(df)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🎙️ Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Dataset Explorer",
        "Biomarker Explorer",
        "Model Performance",
        "Feature Importance",
        "Limitations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project:** Parkinson's Voice Biomarker Analysis  
    **Dataset:** UCI Parkinson's voice dataset  
    **Target:** Healthy vs Parkinson's disease
    """
)


# --------------------------------------------------
# Overview page
# --------------------------------------------------

if page == "Overview":
    section_header(
        "Parkinson's Voice Biomarker Dashboard",
        "Interactive analysis of biomedical voice measurements associated with Parkinson's disease."
    )

    st.markdown(
        """
        <div class="warning-box">
        <b>Important:</b> This dashboard is for educational and portfolio purposes only. 
        It is not a clinical diagnostic tool and should not be used for medical decision-making.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Voice Recordings", df.shape[0])

    with col2:
        metric_card("Extracted Patients", df["patient_id"].nunique())

    with col3:
        metric_card("Biomarkers", len(all_feature_columns))

    with col4:
        metric_card("Target Classes", 2)

    st.write("")

    st.markdown(
        """
        <div class="section-card">
        <h3>Project Goal</h3>
        <p>
        This project investigates whether vocal biomarkers can help distinguish 
        Parkinson's disease recordings from healthy control recordings. The analysis 
        focuses on biomedical voice measurements such as frequency variation, amplitude 
        variation, vocal noise, and nonlinear signal complexity.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        class_counts = df["diagnosis_label"].value_counts().reset_index()
        class_counts.columns = ["Diagnosis", "Recordings"]

        fig = px.bar(
            class_counts,
            x="Diagnosis",
            y="Recordings",
            text="Recordings",
            title="Recording-Level Class Distribution",
            color="Diagnosis",
            color_discrete_map=COLOR_MAP
        )

        fig.update_traces(
            textposition="outside",
            marker_line_color="#e5e7eb",
            marker_line_width=0.7
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(style_plot(fig), use_container_width=True)

    with col2:
        patient_counts = patient_df["diagnosis_label"].value_counts().reset_index()
        patient_counts.columns = ["Diagnosis", "Patients"]

        fig = px.bar(
            patient_counts,
            x="Diagnosis",
            y="Patients",
            text="Patients",
            title="Patient-Level Class Distribution",
            color="Diagnosis",
            color_discrete_map=COLOR_MAP
        )

        fig.update_traces(
            textposition="outside",
            marker_line_color="#e5e7eb",
            marker_line_width=0.7
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown(
        """
        <div class="section-card">
        <h3>Key Findings</h3>
        <ul>
            <li>The dataset is moderately imbalanced toward Parkinson's recordings.</li>
            <li>Repeated recordings per patient require patient-aware validation.</li>
            <li>Nonlinear vocal biomarkers such as <b>PPE</b>, <b>spread1</b>, and <b>spread2</b> were among the strongest features.</li>
            <li>Jitter and shimmer features showed high correlation, suggesting redundancy among related voice measurements.</li>
            <li>Modeling results suggest useful signal, but healthy-class detection remains limited due to small sample size and class imbalance.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Dataset Explorer
# --------------------------------------------------

elif page == "Dataset Explorer":
    section_header(
        "Dataset Explorer",
        "Explore the cleaned Parkinson's voice dataset and patient-level structure."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Rows", df.shape[0])

    with col2:
        metric_card("Columns", df.shape[1])

    with col3:
        metric_card("Patients", df["patient_id"].nunique())

    st.write("")

    st.subheader("Cleaned Dataset")

    with st.expander("View cleaned dataset"):
        st.dataframe(df, use_container_width=True)

    st.subheader("Recordings per Patient")

    recordings_per_patient = (
        df.groupby(["patient_id", "diagnosis_label"])
        .size()
        .reset_index(name="recording_count")
    )

    fig = px.bar(
        recordings_per_patient,
        x="patient_id",
        y="recording_count",
        color="diagnosis_label",
        title="Number of Voice Recordings per Patient",
        labels={
            "patient_id": "Patient ID",
            "recording_count": "Recording Count",
            "diagnosis_label": "Diagnosis"
        },
        color_discrete_map=COLOR_MAP
    )

    st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown(
        """
        <div class="section-card">
        <h3>Why Patient IDs Matter</h3>
        <p>
        The dataset contains multiple recordings from the same patients. 
        This means individual rows are not fully independent. For model evaluation, 
        recordings from the same patient should not appear in both training and testing data.
        Patient-aware splitting helps reduce data leakage.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Biomarker Explorer
# --------------------------------------------------

elif page == "Biomarker Explorer":
    section_header(
        "Biomarker Explorer",
        "Compare healthy and Parkinson's groups across vocal biomarker categories."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_group = st.selectbox(
            "Select biomarker group",
            list(feature_group_map.keys())
        )

    with col2:
        selected_feature = st.selectbox(
            "Select biomarker",
            feature_group_map[selected_group]
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            patient_df,
            x="diagnosis_label",
            y=selected_feature,
            color="diagnosis_label",
            points="all",
            title=f"{selected_feature} by Diagnosis Status",
            labels={
                "diagnosis_label": "Diagnosis",
                selected_feature: selected_feature
            },
            color_discrete_map=COLOR_MAP
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(style_plot(fig), use_container_width=True)

    with col2:
        fig = px.histogram(
            patient_df,
            x=selected_feature,
            color="diagnosis_label",
            marginal="rug",
            barmode="overlay",
            opacity=0.72,
            title=f"Distribution of {selected_feature}",
            labels={
                "diagnosis_label": "Diagnosis"
            },
            color_discrete_map=COLOR_MAP
        )

        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.subheader("Group Summary")

    summary_table = (
        patient_df.groupby("diagnosis_label")[selected_feature]
        .agg(["mean", "median", "std", "min", "max"])
        .round(4)
    )

    st.dataframe(summary_table, use_container_width=True)

    st.markdown(
        """
        <div class="section-card">
        <h3>Interpretation Guide</h3>
        <p>
        Biomarkers with clearer separation between healthy and Parkinson's groups may be more useful 
        for classification. However, visual separation alone does not prove causation or clinical validity. 
        These plots are exploratory and should be interpreted alongside patient-aware cross-validation results.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Model Performance
# --------------------------------------------------

elif page == "Model Performance":
    section_header(
        "Model Performance",
        "Review patient-aware model evaluation results."
    )

    if cv_results is None or model_comparison is None:
        st.warning("Model comparison CSV files were not found. Run the modeling notebook first.")
    else:
        st.subheader("Patient-Aware Cross-Validation Results")

        st.dataframe(cv_results, use_container_width=True)

        cv_metric_options = [
            col for col in cv_results.columns
            if col.endswith("_mean")
        ]

        default_metrics = [
            metric for metric in [
                "accuracy_mean",
                "balanced_accuracy_mean",
                "f1_mean",
                "roc_auc_mean",
                "average_precision_mean"
            ]
            if metric in cv_metric_options
        ]

        selected_metrics = st.multiselect(
            "Select metrics to display",
            cv_metric_options,
            default=default_metrics
        )

        if selected_metrics:
            cv_plot = cv_results.melt(
                id_vars="Model",
                value_vars=selected_metrics,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                cv_plot,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Patient-Aware Cross-Validation Performance",
                color_discrete_sequence=px.colors.qualitative.Set2
            )

            fig.update_yaxes(range=[0, 1.05])

            st.plotly_chart(style_plot(fig), use_container_width=True)

        st.subheader("Single Train/Test Split Results")

        st.dataframe(model_comparison, use_container_width=True)

        st.markdown(
            """
            <div class="section-card">
            <h3>Modeling Takeaway</h3>
            <p>
            Patient-aware cross-validation is more trustworthy than a regular random split because 
            it evaluates whether models generalize to unseen patients. The results show that vocal 
            biomarkers contain useful signal, but performance should be interpreted cautiously because 
            the dataset is small and imbalanced.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

elif page == "Feature Importance":
    section_header(
        "Feature Importance",
        "Explore which vocal biomarkers contributed most to model predictions."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Random Forest Importance",
            "Logistic Regression Coefficients",
            "Feature Group Comparison"
        ]
    )

    with tab1:
        if rf_importance is None:
            st.warning("Random forest feature importance file was not found.")
        else:
            st.dataframe(rf_importance, use_container_width=True)

            fig = px.bar(
                rf_importance.sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest Feature Importance",
                color="Importance",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(style_plot(fig), use_container_width=True)

    with tab2:
        if logistic_coefficients is None:
            st.warning("Logistic regression coefficient file was not found.")
        else:
            st.dataframe(logistic_coefficients, use_container_width=True)

            fig = px.bar(
                logistic_coefficients.sort_values("Coefficient"),
                x="Coefficient",
                y="Feature",
                orientation="h",
                title="Logistic Regression Coefficients",
                color="Coefficient",
                color_continuous_scale="RdBu"
            )

            st.plotly_chart(style_plot(fig), use_container_width=True)

    with tab3:
        if feature_group_results is None:
            st.warning("Feature group comparison file was not found.")
        else:
            st.dataframe(feature_group_results, use_container_width=True)

            metrics = [
                "Balanced Accuracy",
                "F1",
                "ROC-AUC",
                "PR-AUC"
            ]

            available_metrics = [
                metric for metric in metrics
                if metric in feature_group_results.columns
            ]

            selected_metric = st.selectbox(
                "Select metric",
                available_metrics
            )

            fig = px.bar(
                feature_group_results.sort_values(selected_metric, ascending=False),
                x="Feature Group",
                y=selected_metric,
                color="Feature Group",
                title=f"Feature Group Comparison by {selected_metric}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            fig.update_yaxes(range=[0, 1.05])

            st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown(
        """
        <div class="section-card">
        <h3>Feature Importance Takeaway</h3>
        <p>
        The strongest features across EDA and model interpretation include nonlinear biomarkers 
        such as <b>PPE</b>, <b>spread1</b>, and <b>spread2</b>, along with selected frequency, 
        jitter, and shimmer measurements. Because many biomarkers are correlated, individual 
        feature rankings should be interpreted cautiously.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Limitations
# --------------------------------------------------

elif page == "Limitations":
    section_header(
        "Limitations and Next Steps",
        "Important context for interpreting this project responsibly."
    )

    st.markdown(
        """
        <div class="section-card">
        <h3>Limitations</h3>
        <ul>
            <li>The dataset is small, with only 195 recordings and 32 extracted patient IDs.</li>
            <li>The dataset documentation reports 31 individuals, while extracted subject labels show 32 IDs.</li>
            <li>There are multiple recordings per patient, so patient-aware validation is required.</li>
            <li>The dataset is moderately imbalanced toward Parkinson's recordings.</li>
            <li>The model was not externally validated on an independent dataset.</li>
            <li>Many biomarkers are highly correlated, which affects feature importance interpretation.</li>
            <li>The dashboard is educational and should not be used as a medical diagnostic tool.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-card">
        <h3>Future Work</h3>
        <ul>
            <li>Use nested cross-validation for more rigorous hyperparameter tuning.</li>
            <li>Compare patient-level aggregation strategies such as mean, median, and max.</li>
            <li>Evaluate dimensionality reduction with PCA before classification.</li>
            <li>Test model generalization on an external Parkinson's voice dataset.</li>
            <li>Extend the project to the telemonitoring UPDRS dataset for disease severity prediction.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )