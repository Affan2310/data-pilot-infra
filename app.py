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
    st.header("👥 Team Collaboration")
    st.markdown("Manage collaborative data analysis projects and share insights with your team.")
    
    # Project management section
    st.subheader("Project Management")
    
    tab1, tab2, tab3 = st.tabs(["Create Project", "Manage Projects", "Team Analytics"])
    
    with tab1:
        st.write("Create a new collaborative project")
        
        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("Project Name")
            created_by = st.text_input("Your Name")
        
        with col2:
            description = st.text_area("Project Description")
            dataset_id = None
            if st.session_state.data is not None:
                use_current_dataset = st.checkbox("Use current dataset for this project")
                if use_current_dataset:
                    dataset_id = 1  # Simplified for demo
        
        if st.button("Create Project") and project_name and created_by:
            result = collaboration_manager.create_project(project_name, description, created_by, dataset_id)
            
            if result['success']:
                st.success(f"Project '{project_name}' created successfully!")
                st.write(f"Project ID: {result['project_id']}")
            else:
                st.error(result['error'])
    
    with tab2:
        st.write("Manage existing projects")
        
        # Demo project for display
        demo_project_id = "demo123"
        
        col1, col2 = st.columns(2)
        with col1:
            project_id = st.text_input("Project ID", value=demo_project_id)
            member_name = st.text_input("Team Member Name")
        
        with col2:
            permission_level = st.selectbox("Permission Level", ["viewer", "contributor", "admin"])
        
        if st.button("Add Team Member") and project_id and member_name:
            result = collaboration_manager.add_team_member(project_id, member_name, permission_level)
            
            if result['success']:
                st.success(result['message'])
            else:
                st.error(result['error'])
        
        # Share insights section
        st.subheader("Share Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            insight_title = st.text_input("Insight Title")
            user_name = st.text_input("Your Name", key="insight_user")
        
        with col2:
            insight_type = st.selectbox("Insight Type", ["general", "data_quality", "pattern", "recommendation"])
            insight_content = st.text_area("Insight Content")
        
        if st.button("Share Insight") and project_id and insight_title and insight_content:
            result = collaboration_manager.share_insight(project_id, user_name, insight_title, insight_content, insight_type)
            
            if result['success']:
                st.success(result['message'])
            else:
                st.error(result['error'])
    
    with tab3:
        st.write("Team analytics and project overview")
        
        project_id_analytics = st.text_input("Project ID for Analytics", value="demo123")
        
        if st.button("Generate Analytics") and project_id_analytics:
            # Get project summary
            summary = collaboration_manager.get_project_summary(project_id_analytics)
            
            if 'error' not in summary:
                st.subheader("Project Overview")
                
                # Project info
                proj_info = summary['project_info']
                st.write(f"**Project:** {proj_info['name']}")
                st.write(f"**Created by:** {proj_info['created_by']}")
                st.write(f"**Status:** {proj_info['status']}")
                
                # Statistics
                stats = summary['statistics']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Team Size", stats['team_size'])
                with col2:
                    st.metric("Total Analyses", stats['total_analyses'])
                with col3:
                    st.metric("Shared Insights", stats['total_insights'])
                
                # Team members
                if summary['team_members']:
                    st.subheader("Team Members")
                    for member in summary['team_members']:
                        st.write(f"- {member}")
                
                # Top contributors
                if summary['top_contributors']:
                    st.subheader("Top Contributors")
                    for contributor, count in summary['top_contributors']:
                        st.write(f"- {contributor}: {count} analyses")
            
            else:
                st.warning("Project not found or no data available for analytics")
    
    # Sample collaboration features demo
    st.subheader("Collaboration Features Demo")
    
    with st.expander("View Sample Team Dashboard"):
        st.write("**Sample Project: Market Analysis Team**")
        
        # Mock data for demonstration
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Projects", "3")
        with col2:
            st.metric("Team Members", "5")
        with col3:
            st.metric("Analyses This Week", "12")
        with col4:
            st.metric("Collaboration Score", "85/100")
        
        st.write("**Recent Activity:**")
        st.write("- Sarah completed EDA analysis on customer data")
        st.write("- Mike shared insights about seasonal trends")
        st.write("- Lisa uploaded new product sales dataset")
        st.write("- Team completed quarterly forecast model")
        
        st.write("**Shared Insights:**")
        st.write("1. **Customer Segmentation Patterns** - Data shows 3 distinct customer groups")
        st.write("2. **Seasonal Sales Trends** - Peak sales occur in Q4 with 40% increase")
        st.write("3. **Data Quality Issues** - Missing values in 15% of transaction records")

if __name__ == "__main__":
    main()