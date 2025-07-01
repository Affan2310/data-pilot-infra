import streamlit as st
import pandas as pd
import numpy as np
import json
from io import BytesIO
from PIL import Image
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from utils.data_profiler import DataProfiler
from utils.eda_generator import EDAGenerator
from utils.ml_builder import MLBuilder
from utils.computer_vision import ComputerVisionAnalyzer
from utils.feature_engineering import AdvancedFeatureEngineer
from utils.time_series_analyzer import TimeSeriesAnalyzer
from utils.report_generator import ReportGenerator
from utils.collaboration_features import CollaborationManager
from utils.dummy_data import generate_dummy_classification_data, generate_dummy_regression_data


st.set_page_config(
    page_title="DataSciPilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'data' not in st.session_state:
    st.session_state.data = None
if 'profiling_results' not in st.session_state:
    st.session_state.profiling_results = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

def main():
    st.title("🚀 DataSciPilot")
    st.markdown("### Lightweight Data Analysis & ML Model Building Platform")
    st.markdown("---")

    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Upload", "Data Profiling", "EDA & Visualization", 
         "ML Model Building", "Computer Vision", 
         "Feature Engineering", "Time Series Analysis", 
         "Download Reports", "Team Collaboration"]
    )

    data_profiler = DataProfiler()
    eda_generator = EDAGenerator()
    ml_builder = MLBuilder()
    cv_analyzer = ComputerVisionAnalyzer()
    feature_engineer = AdvancedFeatureEngineer()
    ts_analyzer = TimeSeriesAnalyzer()
    report_generator = ReportGenerator()
    collaboration_manager = CollaborationManager()

    if page == "Data Upload":
        data_upload_page(data_profiler)
    elif page == "Data Profiling":
        data_profiling_page(data_profiler)
    elif page == "EDA & Visualization":
        eda_page(eda_generator)
    elif page == "ML Model Building":
        ml_page(ml_builder)
    elif page == "Computer Vision":
        computer_vision_page(cv_analyzer)
    elif page == "Feature Engineering":
        feature_engineering_page(feature_engineer)
    elif page == "Time Series Analysis":
        time_series_page(ts_analyzer)
    elif page == "Download Reports":
        download_page(report_generator)
    elif page == "Team Collaboration":
        collaboration_page(collaboration_manager)


def data_upload_page(data_profiler):
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or JSON", type=['csv', 'json'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                data = json.load(uploaded_file)
                df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
            st.session_state.data = df
            st.success(f"✅ Loaded {uploaded_file.name}")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
    
    st.markdown("### Or generate dummy data:")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Dummy Classification Data"):
            st.session_state.data = generate_dummy_classification_data()
            st.success("✅ Dummy classification dataset loaded!")
            st.dataframe(st.session_state.data.head())
    
    with col2:
        if st.button("Generate Dummy Regression Data"):
            st.session_state.data = generate_dummy_regression_data()
            st.success("✅ Dummy regression dataset loaded!")
            st.dataframe(st.session_state.data.head())
    
    # Ensure target selection doesn't pre-fill unintended columns later
    st.session_state.target_selected = None


def data_profiling_page(profiler):
    st.header("📊 Data Profiling")
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    profile = profiler.generate_profile(st.session_state.data)
    report_text = profiler.generate_human_friendly_report(profile)
    st.code(report_text, language="text")


def eda_page(eda_generator):
    st.header("📈 EDA")
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        col = st.selectbox("Select numeric column", numeric_cols)
        if st.button("Show Distribution"):
            fig = eda_generator.create_distribution_plot(st.session_state.data, col)
            st.plotly_chart(fig)


def ml_page(ml_builder):
    st.header("🤖 Machine Learning Builder")

    if st.session_state.data is None:
        st.warning("Please upload data or generate dummy data first.")
        return

    target = _select_target_column(st.session_state.data)
    problem_type = st.radio(
        "Problem type", 
        ['classification', 'regression'], 
        help="Classification = predict categories, Regression = predict numeric values"
    )

    if st.button("Run ML Builder"):
        if not target:
            st.warning("⚠️ Please select a valid target column before running model building.")
            return

        feature_cols = [col for col in st.session_state.data.columns if col != target]
        _run_model_building(ml_builder, st.session_state.data, target, feature_cols, problem_type)


def _select_target_column(df):
    """
    Render a selectbox for target column selection.
    Forces the user to pick one.
    """
    cols = df.columns.tolist()
    selection = st.selectbox(
        "Select target column (required):",
        ["-- Select target column --"] + cols
    )
    return selection if selection != "-- Select target column --" else None


def _run_model_building(ml_builder, df, target, features, problem_type):
    """
    Run model building and handle output display.
    """
    try:
        results = ml_builder.build_models(df, target, features, problem_type)

        if 'error' in results:
            st.error(results['error'])
        else:
            _display_model_results(results)

    except Exception as e:
        st.error(f"Model building failed: {str(e)}")


def _display_model_results(results):
    """
    Show best model + all model details.
    """
    best = results['best_model']
    st.success(f"✅ Best Model: {best['model_name']}")
    st.write(f"Test Score: {best['test_score']:.4f}")
    st.write(f"CV Score: {best['cv_score']:.4f}")

    st.write("🔍 All Model Results")
    for r in results['all_results']:
        st.subheader(r['model_name'])
        st.write(f"Test Score: {r['test_score']:.4f}")
        st.write(f"CV Score: {r['cv_score']:.4f}")
        if r['feature_importance']:
            st.write("Feature Importance")
            st.json(r['feature_importance'])


def computer_vision_page(cv_analyzer):
    st.header("🔍 Computer Vision")
    st.write("Upload image for defect detection.")

    image_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
    if image_file:
        image_bytes = image_file.read()
        result = cv_analyzer.analyze_image_defects(image_bytes, image_file.name)

        # Show human-friendly report
        st.subheader("📄 Defect Analysis Summary")
        report_text = cv_analyzer.generate_human_friendly_report(result)
        st.text(report_text)

        # Optionally show raw JSON for advanced users
        with st.expander("🔎 View Raw JSON Result"):
            st.json(result)

        # Show annotated image
        st.subheader("🖼 Annotated Image")
        annotated_bytes = cv_analyzer.create_annotated_image(image_bytes, result)
        st.image(Image.open(BytesIO(annotated_bytes)), caption="Defect Annotation")


def feature_engineering_page(feature_engineer):
    st.header("⚙️ Feature Engineering")
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return

    target = st.selectbox("Select target column", st.session_state.data.columns)

    if st.button("Run Feature Engineering"):
        results = feature_engineer.auto_engineer_features(st.session_state.data, target, 50)

        # Generate human-friendly report
        report_text = feature_engineer.generate_human_friendly_feature_report(results)

        # Display the report
        st.code(report_text, language="text")

        # Optionally show a sample of engineered dataset
        if isinstance(results.get("engineered_dataset"), pd.DataFrame):
            st.subheader("Sample of Engineered Dataset")
            st.dataframe(results["engineered_dataset"].head())

        # Optionally show feature importance summary if available
        if results.get("feature_importance"):
            st.subheader("Feature Importance Summary")
            summary = feature_engineer.get_feature_importance_summary(results["feature_importance"])
            st.json(summary)


def time_series_page(ts_analyzer):
    st.header("📈 Time Series Analysis")

    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return

    date_col = st.selectbox("Select date column", st.session_state.data.columns)
    value_col = st.selectbox("Select value column", st.session_state.data.select_dtypes(include=[np.number]).columns)

    if st.button("Run Time Series Analysis"):
        results = ts_analyzer.analyze_time_series(st.session_state.data, date_col, value_col)

        # Human-friendly summary
        st.subheader("📄 Time Series Summary Report")
        report_text = ts_analyzer.generate_human_friendly_time_series_report(results)
        # st.markdown(f"```text\n{report_text}\n```")
        st.code(report_text, language='text')

        # Raw JSON in expandable section
        with st.expander("🔎 View Raw JSON Result"):
            st.json(results)


def download_page(report_generator):
    st.header("📥 Download Reports")
    if st.session_state.profiling_results:
        report = report_generator.generate_profile_report(st.session_state.data, st.session_state.profiling_results)
        st.download_button("Download JSON Report", json.dumps(report, indent=2), "report.json", "application/json")

def collaboration_page(collaboration_manager):
    st.header("👥 Collaboration Features (Demo)")
    st.write("Collaboration tools placeholder.")

if __name__ == "__main__":
    main()