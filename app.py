import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import utility modules
from utils.data_profiler import DataProfiler
from utils.eda_generator import EDAGenerator
from utils.ml_builder import MLBuilder
from utils.nlp_insights import NLPInsights
from utils.report_generator import ReportGenerator

# Configure page
st.set_page_config(
    page_title="DataSciPilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'profiling_results' not in st.session_state:
    st.session_state.profiling_results = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

def main():
    st.title("🚀 DataSciPilot")
    st.markdown("### AI-Powered Data Analysis & ML Model Building Platform")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Upload", "Data Profiling", "EDA & Visualization", "ML Model Building", "AI Insights", "Download Reports"]
    )
    
    # Initialize utility classes
    data_profiler = DataProfiler()
    eda_generator = EDAGenerator()
    ml_builder = MLBuilder()
    nlp_insights = NLPInsights()
    report_generator = ReportGenerator()
    
    if page == "Data Upload":
        data_upload_page(data_profiler)
    elif page == "Data Profiling":
        data_profiling_page(data_profiler)
    elif page == "EDA & Visualization":
        eda_page(eda_generator)
    elif page == "ML Model Building":
        ml_page(ml_builder)
    elif page == "AI Insights":
        ai_insights_page(nlp_insights)
    elif page == "Download Reports":
        download_page(report_generator)

def data_upload_page(data_profiler):
    st.header("📁 Data Upload")
    
    # Sample data option
    st.subheader("Try with Sample Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Iris Dataset", type="secondary"):
            try:
                df = pd.read_csv("sample_data/iris.csv")
                st.session_state.data = df
                st.success("Iris dataset loaded successfully!")
                st.dataframe(df.head())
            except FileNotFoundError:
                st.error("Sample dataset not found. Please upload your own data.")
    
    with col2:
        if st.button("Load House Prices Dataset", type="secondary"):
            try:
                df = pd.read_csv("sample_data/house_prices.csv")
                st.session_state.data = df
                st.success("House prices dataset loaded successfully!")
                st.dataframe(df.head())
            except FileNotFoundError:
                st.error("Sample dataset not found. Please upload your own data.")
    
    st.markdown("---")
    
    # File upload
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'json'],
        help="Upload CSV or JSON files for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
            
            st.session_state.data = df
            st.success(f"Successfully loaded {uploaded_file.name}")
            
            # Display basic info
            st.subheader("Dataset Preview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(10))
            
            # Basic statistics
            st.subheader("Quick Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def data_profiling_page(data_profiler):
    st.header("📊 Data Profiling")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload section.")
        return
    
    df = st.session_state.data
    
    if st.button("Generate Data Profile", type="primary"):
        with st.spinner("Analyzing data..."):
            profile_results = data_profiler.generate_profile(df)
            st.session_state.profiling_results = profile_results
    
    if st.session_state.profiling_results:
        results = st.session_state.profiling_results
        
        # Data Overview
        st.subheader("📋 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Shape", f"{results['shape'][0]} × {results['shape'][1]}")
        with col2:
            st.metric("Missing Values", f"{results['missing_percentage']:.1f}%")
        with col3:
            st.metric("Duplicate Rows", results['duplicates'])
        
        # Data Types
        st.subheader("🏷️ Data Types")
        dtype_df = pd.DataFrame(list(results['dtypes'].items()), columns=['Column', 'Data Type'])
        st.dataframe(dtype_df, use_container_width=True)
        
        # Missing Values Analysis
        if results['missing_values']:
            st.subheader("❌ Missing Values Analysis")
            missing_df = pd.DataFrame(list(results['missing_values'].items()), 
                                    columns=['Column', 'Missing Count'])
            missing_df['Missing Percentage'] = (missing_df['Missing Count'] / len(df)) * 100
            st.dataframe(missing_df, use_container_width=True)
        
        # Data Quality Issues
        st.subheader("⚠️ Data Quality Assessment")
        quality_issues = results['quality_issues']
        
        if quality_issues['high_cardinality']:
            st.warning("High Cardinality Columns: " + ", ".join(quality_issues['high_cardinality']))
        
        if quality_issues['potential_outliers']:
            st.warning("Columns with Potential Outliers: " + ", ".join(quality_issues['potential_outliers']))
        
        if not quality_issues['high_cardinality'] and not quality_issues['potential_outliers']:
            st.success("No major data quality issues detected!")
        
        # Cleaning Suggestions
        st.subheader("🧹 Data Cleaning Suggestions")
        suggestions = results['cleaning_suggestions']
        
        for suggestion in suggestions:
            st.info(f"💡 {suggestion}")

def eda_page(eda_generator):
    st.header("📈 Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload section.")
        return
    
    df = st.session_state.data
    
    # EDA Options
    st.subheader("Select Analysis Type")
    analysis_type = st.selectbox(
        "Choose analysis:",
        ["Distribution Analysis", "Correlation Analysis", "Missing Values Heatmap", "Custom Visualization"]
    )
    
    if analysis_type == "Distribution Analysis":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
            if st.button("Generate Distribution Plot"):
                fig = eda_generator.create_distribution_plot(df, selected_col)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found for distribution analysis.")
    
    elif analysis_type == "Correlation Analysis":
        if st.button("Generate Correlation Heatmap"):
            fig = eda_generator.create_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation analysis.")
    
    elif analysis_type == "Missing Values Heatmap":
        if st.button("Generate Missing Values Heatmap"):
            fig = eda_generator.create_missing_values_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Custom Visualization":
        st.subheader("Custom Plot Builder")
        plot_type = st.selectbox("Plot Type:", ["Scatter Plot", "Box Plot", "Bar Chart", "Histogram"])
        
        if plot_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis:", numeric_cols)
                y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col])
                if st.button("Create Scatter Plot"):
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column:", numeric_cols)
                if st.button("Create Box Plot"):
                    fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)

def ml_page(ml_builder):
    st.header("🤖 ML Model Building")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload section.")
        return
    
    df = st.session_state.data
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    # Target variable selection
    target_col = st.selectbox("Select target variable:", df.columns.tolist())
    
    # Problem type detection
    if df[target_col].dtype == 'object' or df[target_col].nunique() <= 10:
        problem_type = "classification"
        st.info("🎯 Detected: Classification Problem")
    else:
        problem_type = "regression"
        st.info("📊 Detected: Regression Problem")
    
    # Feature selection
    feature_cols = st.multiselect(
        "Select features (leave empty for auto-selection):",
        [col for col in df.columns if col != target_col],
        default=[]
    )
    
    if not feature_cols:
        # Auto-select numeric features
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        st.info(f"Auto-selected numeric features: {', '.join(feature_cols)}")
    
    # Model building
    if st.button("Build ML Models", type="primary"):
        if not feature_cols:
            st.error("No features selected for model building.")
            return
        
        with st.spinner("Building ML models..."):
            try:
                results = ml_builder.build_models(df, target_col, feature_cols, problem_type)
                st.session_state.ml_results = results
                
                # Display results
                st.subheader("🏆 Model Performance Results")
                
                # Create results dataframe
                results_df = pd.DataFrame(results['model_performance'])
                st.dataframe(results_df, use_container_width=True)
                
                # Best model highlight
                if problem_type == "classification":
                    best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
                    best_score = results_df['Accuracy'].max()
                    st.success(f"🥇 Best Model: {best_model} (Accuracy: {best_score:.3f})")
                else:
                    best_model = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
                    best_score = results_df['R² Score'].max()
                    st.success(f"🥇 Best Model: {best_model} (R² Score: {best_score:.3f})")
                
                # Feature importance (if available)
                if 'feature_importance' in results and results['feature_importance']:
                    st.subheader("📊 Feature Importance")
                    importance_df = pd.DataFrame(
                        list(results['feature_importance'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Importance', y='Feature', 
                               orientation='h', title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error building models: {str(e)}")

def ai_insights_page(nlp_insights):
    st.header("🧠 AI-Powered Insights")
    
    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload section.")
        return
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.info("💡 This feature requires an OpenAI API key to generate natural language insights.")
        return
    
    df = st.session_state.data
    
    st.subheader("Generate AI Insights")
    
    # Insight type selection
    insight_type = st.selectbox(
        "Select insight type:",
        ["Data Summary", "Pattern Analysis", "Anomaly Detection", "Business Recommendations"]
    )
    
    if st.button("Generate AI Insights", type="primary"):
        with st.spinner("Generating AI insights..."):
            try:
                if insight_type == "Data Summary":
                    insights = nlp_insights.generate_data_summary(df)
                elif insight_type == "Pattern Analysis":
                    insights = nlp_insights.analyze_patterns(df)
                elif insight_type == "Anomaly Detection":
                    insights = nlp_insights.detect_anomalies(df)
                elif insight_type == "Business Recommendations":
                    insights = nlp_insights.generate_recommendations(df)
                
                st.subheader(f"🎯 {insight_type} Results")
                st.write(insights)
                
                # Additional context for ML results
                if st.session_state.ml_results:
                    st.subheader("🤖 ML Model Insights")
                    ml_insights = nlp_insights.analyze_ml_results(st.session_state.ml_results)
                    st.write(ml_insights)
                
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
    
    # Manual insight generation
    st.subheader("Custom Analysis")
    custom_question = st.text_area("Ask a specific question about your data:")
    
    if st.button("Get Custom Insight") and custom_question:
        with st.spinner("Analyzing..."):
            try:
                custom_insight = nlp_insights.answer_custom_question(df, custom_question)
                st.write("**Answer:**")
                st.write(custom_insight)
            except Exception as e:
                st.error(f"Error generating custom insight: {str(e)}")

def download_page(report_generator):
    st.header("📥 Download Reports")
    
    if st.session_state.data is None:
        st.warning("Please upload data first and run some analysis.")
        return
    
    st.subheader("Available Reports")
    
    # Data Profile Report
    if st.session_state.profiling_results:
        if st.button("Download Data Profile Report", type="secondary"):
            report = report_generator.generate_profile_report(
                st.session_state.data, 
                st.session_state.profiling_results
            )
            st.download_button(
                label="📄 Download Profile Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # ML Results Report
    if st.session_state.ml_results:
        if st.button("Download ML Results Report", type="secondary"):
            report = report_generator.generate_ml_report(st.session_state.ml_results)
            st.download_button(
                label="📊 Download ML Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Complete Analysis Report
    st.subheader("Complete Analysis Package")
    if st.button("Generate Complete Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            complete_report = report_generator.generate_complete_report(
                st.session_state.data,
                st.session_state.profiling_results,
                st.session_state.ml_results
            )
            
            st.download_button(
                label="📦 Download Complete Analysis (JSON)",
                data=json.dumps(complete_report, indent=2, default=str),
                file_name=f"complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Dataset Export
    st.subheader("Export Processed Data")
    if st.button("Export Current Dataset", type="secondary"):
        csv_buffer = io.StringIO()
        st.session_state.data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="💾 Download Dataset (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
