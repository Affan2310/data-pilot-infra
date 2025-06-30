# DataSciPilot - AI-Powered Data Analysis Platform

## Overview

DataSciPilot is a comprehensive data science platform built with Streamlit that provides automated data analysis, machine learning model building, and AI-powered insights generation. The application serves as an all-in-one solution for data scientists and analysts to quickly explore datasets, build models, and generate comprehensive reports.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with interactive UI
- **Layout**: Wide layout with expandable sidebar navigation
- **State Management**: Streamlit session state for maintaining data and results across interactions
- **Visualization**: Plotly Express and Plotly Graph Objects for interactive charts and plots

### Backend Architecture
- **Modular Design**: Utility-based architecture with separate modules for different functionalities
- **Session Management**: Stateful design using Streamlit's session state to persist user data
- **Database Integration**: PostgreSQL database for persistent storage of datasets and analysis results
- **Processing Pipeline**: Sequential workflow from data upload → profiling → EDA → ML → insights → reporting

### Key Components

1. **Data Profiler** (`utils/data_profiler.py`)
   - Comprehensive data quality assessment
   - Statistical analysis and missing value detection
   - Data type inference and memory usage analysis
   - Quality issue identification and cleaning suggestions

2. **EDA Generator** (`utils/eda_generator.py`)
   - Interactive visualization creation using Plotly
   - Distribution plots, correlation matrices, and statistical plots
   - Automated chart generation based on data types
   - Multi-subplot layouts for comprehensive visual analysis

3. **ML Builder** (`utils/ml_builder.py`)
   - Automated machine learning pipeline
   - Support for both classification and regression tasks
   - Multiple algorithm comparison (Random Forest, Logistic Regression, SVM, KNN)
   - Automated preprocessing with StandardScaler and imputation
   - Cross-validation and model evaluation metrics

4. **NLP Insights** (`utils/nlp_insights.py`)
   - OpenAI API integration for natural language insights
   - Automated data summary generation
   - AI-powered interpretation of analysis results
   - Environment variable configuration for API keys

5. **Report Generator** (`utils/report_generator.py`)
   - Comprehensive report generation in multiple formats
   - Structured output with timestamps and metadata
   - Integration of all analysis components into unified reports

6. **Database Manager** (`utils/database_manager.py`)
   - PostgreSQL database integration for persistent storage
   - Dataset metadata and analysis results tracking
   - Historical analysis session management
   - Dataset versioning and retrieval capabilities

## Data Flow

1. **Data Upload**: Users upload CSV/Excel files through Streamlit interface
2. **Data Profiling**: Automatic statistical analysis and quality assessment
3. **Exploratory Data Analysis**: Interactive visualization generation
4. **Model Building**: Automated ML pipeline with multiple algorithms
5. **AI Insights**: Natural language interpretation of results
6. **Report Generation**: Comprehensive downloadable reports

## External Dependencies

### Required Python Packages
- **Core**: `streamlit`, `pandas`, `numpy`
- **Visualization**: `plotly`
- **Machine Learning**: `scikit-learn`, `scipy`
- **AI Integration**: `openai`
- **Utilities**: `datetime`, `json`, `io`, `os`

### API Integrations
- **OpenAI API**: For natural language insights generation
- **Configuration**: Requires `OPENAI_API_KEY` environment variable

## Deployment Strategy

### Development Environment
- Streamlit development server for local testing
- Modular architecture allows for easy component testing
- Environment variable configuration for API keys

### Production Considerations
- Streamlit Cloud or container-based deployment
- Secure API key management through environment variables
- Session state management for multi-user scenarios
- Memory optimization for large dataset handling

### Scalability Features
- Modular design allows for easy feature addition
- Stateless utility classes for concurrent processing
- Configurable model parameters and processing options

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- June 30, 2025: Initial setup and complete DataSciPilot application deployment
- June 30, 2025: Configured all dependencies in pyproject.toml (modern Python standard)
- June 30, 2025: Integrated OpenAI API for AI-powered insights functionality
- June 30, 2025: Added PostgreSQL database integration for persistent storage
- June 30, 2025: Implemented database history page with dataset management features
- June 30, 2025: Application successfully running on port 5000, ready for portfolio demos