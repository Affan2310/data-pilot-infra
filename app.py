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
from utils.database_manager import DatabaseManager

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
        ["Data Upload", "Data Profiling", "EDA & Visualization", "ML Model Building", "AI Insights", "Download Reports", "Database History"]
    )
    
    # Initialize utility classes
    data_profiler = DataProfiler()
    eda_generator = EDAGenerator()
    ml_builder = MLBuilder()
    nlp_insights = NLPInsights()
    report_generator = ReportGenerator()
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        db_manager = None
    
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
    elif page == "Database History":
        database_history_page(db_manager)

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

def database_history_page(db_manager):
    st.header("🗄️ Database History")
    
    if not db_manager:
        st.error("Database connection not available.")
        return
    
    try:
        # Database statistics
        st.subheader("📊 Database Statistics")
        stats = db_manager.get_database_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Datasets", stats['total_datasets'])
        with col2:
            st.metric("Total Analyses", stats['total_analyses'])
        with col3:
            st.metric("Datasets Today", stats['datasets_today'])
        with col4:
            st.metric("Analyses Today", stats['analyses_today'])
        
        st.markdown("---")
        
        # Dataset history
        st.subheader("📁 Dataset History")
        
        if st.button("Refresh History"):
            st.rerun()
        
        history = db_manager.get_datasets_history()
        
        if not history:
            st.info("No datasets found in database. Upload and analyze data to see history here.")
            return
        
        # Display datasets in a table
        history_df = pd.DataFrame(history)
        
        # Format the dataframe for better display
        if 'upload_timestamp' in history_df.columns:
            history_df['upload_timestamp'] = pd.to_datetime(history_df['upload_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Add status indicators
        status_indicators = []
        for _, row in history_df.iterrows():
            status = []
            if row.get('has_profiling', False):
                status.append("📊")
            if row.get('has_ml_results', False):
                status.append("🤖")
            if row.get('has_ai_insights', False):
                status.append("🧠")
            status_indicators.append(" ".join(status) if status else "❌")
        
        history_df['Analysis Status'] = status_indicators
        
        # Select columns for display
        display_cols = ['id', 'name', 'upload_timestamp', 'rows', 'columns', 'file_type', 'Analysis Status']
        display_df = history_df[display_cols].copy()
        display_df.columns = ['ID', 'Dataset Name', 'Upload Time', 'Rows', 'Columns', 'Type', 'Analysis Status']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Dataset details section
        st.subheader("🔍 Dataset Details")
        
        if len(history) > 0:
            dataset_names = [f"{row['id']} - {row['name']}" for row in history]
            selected_dataset = st.selectbox("Select dataset for details:", ["Select a dataset..."] + dataset_names)
            
            if selected_dataset != "Select a dataset...":
                dataset_id = int(selected_dataset.split(" - ")[0])
                
                # Load dataset details
                with st.spinner("Loading dataset details..."):
                    details = db_manager.get_dataset_details(dataset_id)
                    
                    if details:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Dataset:** {details['name']}")
                            st.write(f"**Upload Time:** {details['upload_timestamp']}")
                            st.write(f"**Dimensions:** {details['rows']} rows × {details['columns']} columns")
                            st.write(f"**File Type:** {details['file_type']}")
                            st.write(f"**Memory Usage:** {details['memory_usage_mb']:.2f} MB")
                            st.write(f"**Missing Data:** {details['missing_percentage']:.1f}%")
                            st.write(f"**Duplicates:** {details['duplicates']}")
                        
                        with col2:
                            # Load dataset from database
                            if st.button("Load Dataset", key=f"load_{dataset_id}"):
                                with st.spinner("Loading dataset from database..."):
                                    df = db_manager.load_dataset_from_db(dataset_id)
                                    if df is not None:
                                        st.session_state.data = df
                                        st.success("Dataset loaded successfully!")
                                        st.dataframe(df.head())
                                    else:
                                        st.error("Failed to load dataset from database.")
                            
                            # Delete dataset
                            if st.button("🗑️ Delete Dataset", key=f"delete_{dataset_id}"):
                                if st.session_state.get(f'confirm_delete_{dataset_id}', False):
                                    with st.spinner("Deleting dataset..."):
                                        if db_manager.delete_dataset(dataset_id):
                                            st.success("Dataset deleted successfully!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to delete dataset.")
                                else:
                                    st.session_state[f'confirm_delete_{dataset_id}'] = True
                                    st.warning("Click again to confirm deletion.")
                        
                        # Show analysis results if available
                        if details.get('profile_results'):
                            with st.expander("📊 Profiling Results"):
                                st.json(details['profile_results'])
                        
                        if details.get('ml_results'):
                            with st.expander("🤖 ML Results"):
                                st.json(details['ml_results'])
                        
                        if details.get('ai_insights'):
                            with st.expander("🧠 AI Insights"):
                                st.write(details['ai_insights'])
                    
                    else:
                        st.error("Dataset details not found.")
        
        # Save current dataset to database
        st.markdown("---")
        st.subheader("💾 Save Current Dataset")
        
        if st.session_state.data is not None:
            dataset_name = st.text_input("Dataset name:", value="My Dataset")
            
            if st.button("Save to Database"):
                if dataset_name.strip():
                    with st.spinner("Saving dataset to database..."):
                        try:
                            dataset_id = db_manager.save_dataset(dataset_name, st.session_state.data)
                            
                            # Save analysis results if available
                            if st.session_state.profiling_results:
                                db_manager.update_dataset_analysis(dataset_id, 'profiling', st.session_state.profiling_results)
                            
                            if st.session_state.ml_results:
                                db_manager.update_dataset_analysis(dataset_id, 'ml', st.session_state.ml_results)
                            
                            st.success(f"Dataset saved successfully! Database ID: {dataset_id}")
                        except Exception as e:
                            st.error(f"Error saving dataset: {e}")
                else:
                    st.error("Please enter a dataset name.")
        else:
            st.info("No dataset loaded. Upload data first to save to database.")
    
    except Exception as e:
        st.error(f"Database error: {e}")

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
