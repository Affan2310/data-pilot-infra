import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from openai import OpenAI

class NLPInsights:
    """Generate natural language insights using OpenAI API."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def _check_api_availability(self):
        """Check if OpenAI API is available."""
        if not self.client:
            raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    def generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate natural language summary of the dataset."""
        self._check_api_availability()
        
        # Prepare data summary
        summary_stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': dict(df.dtypes.astype(str)),
            'missing_values': dict(df.isnull().sum()),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary_stats['numeric_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary_stats['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'most_frequent': str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else 'N/A'
            }
        
        prompt = f"""
        You are a data scientist analyzing a dataset. Provide a comprehensive, professional summary of the following dataset information in natural language. Make it informative and actionable for business stakeholders.

        Dataset Information:
        {json.dumps(summary_stats, indent=2)}

        Please provide:
        1. Overview of the dataset structure and size
        2. Key characteristics of the data
        3. Data quality observations
        4. Interesting patterns or insights
        5. Recommendations for further analysis

        Keep the response professional and suitable for a business presentation.
        """
        
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist providing insights on datasets."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating data summary: {str(e)}"
    
    def analyze_patterns(self, df: pd.DataFrame) -> str:
        """Analyze patterns and relationships in the data."""
        self._check_api_availability()
        
        # Calculate correlations for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = {}
        
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            # Find strong correlations (> 0.7 or < -0.7)
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            correlations['strong_correlations'] = strong_corr
        
        # Analyze distributions
        distribution_info = {}
        for col in numeric_df.columns:
            skewness = float(df[col].skew())
            kurtosis = float(df[col].kurtosis())
            distribution_info[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'distribution_type': 'normal' if abs(skewness) < 0.5 else 'skewed'
            }
        
        analysis_data = {
            'correlations': correlations,
            'distributions': distribution_info,
            'data_shape': df.shape
        }
        
        prompt = f"""
        As a data scientist, analyze the following statistical patterns in the dataset and provide insights:

        Pattern Analysis:
        {json.dumps(analysis_data, indent=2)}

        Please provide:
        1. Key relationships and correlations found
        2. Distribution characteristics and what they indicate
        3. Potential data relationships that warrant further investigation
        4. Statistical patterns that could impact modeling
        5. Recommendations for feature engineering or data transformation

        Focus on actionable insights for data science workflows.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert statistician analyzing data patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing patterns: {str(e)}"
    
    def detect_anomalies(self, df: pd.DataFrame) -> str:
        """Detect and explain potential anomalies in the data."""
        self._check_api_availability()
        
        anomaly_info = {}
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                anomaly_info[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(df)) * 100,
                    'extreme_values': {
                        'min_outliers': list(df[df[col] < lower_bound][col].head(3).values),
                        'max_outliers': list(df[df[col] > upper_bound][col].head(3).values)
                    }
                }
        
        # Check for unusual patterns
        pattern_info = {}
        
        # Check for columns with too many unique values
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                pattern_info[col] = f"High cardinality: {unique_ratio:.2%} unique values"
        
        # Check for columns with too few unique values
        for col in df.columns:
            if df[col].nunique() == 1:
                pattern_info[col] = "Constant column - no variation"
        
        analysis_data = {
            'outliers': anomaly_info,
            'patterns': pattern_info,
            'total_rows': len(df)
        }
        
        prompt = f"""
        As a data quality expert, analyze the following anomaly detection results and provide insights:

        Anomaly Analysis:
        {json.dumps(analysis_data, indent=2)}

        Please provide:
        1. Assessment of outliers found and their potential causes
        2. Evaluation of unusual patterns in the data
        3. Recommendations for handling anomalies
        4. Potential impact on data analysis and modeling
        5. Data quality improvement suggestions

        Focus on practical recommendations for data preprocessing and quality assurance.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data quality expert specializing in anomaly detection."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error detecting anomalies: {str(e)}"
    
    def generate_recommendations(self, df: pd.DataFrame) -> str:
        """Generate business recommendations based on data analysis."""
        self._check_api_availability()
        
        # Gather comprehensive data insights
        data_profile = {
            'shape': df.shape,
            'data_types': dict(df.dtypes.astype(str)),
            'missing_data': dict(df.isnull().sum()),
            'sample_data': df.head(3).to_dict('records')
        }
        
        # Add statistical insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data_profile['numeric_insights'] = {
                'columns': list(numeric_cols),
                'summary_stats': df[numeric_cols].describe().to_dict()
            }
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            data_profile['categorical_insights'] = {}
            for col in categorical_cols:
                data_profile['categorical_insights'][col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_values': list(df[col].value_counts().head(5).to_dict().keys())
                }
        
        prompt = f"""
        As a business intelligence consultant, analyze this dataset and provide strategic recommendations:

        Dataset Profile:
        {json.dumps(data_profile, indent=2, default=str)}

        Please provide:
        1. Business value opportunities from this data
        2. Key metrics and KPIs that could be derived
        3. Potential use cases and applications
        4. Data enrichment suggestions
        5. Strategic recommendations for data-driven decision making
        6. Next steps for analysis and implementation

        Focus on actionable business insights and practical recommendations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a business intelligence consultant providing strategic data insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def analyze_ml_results(self, ml_results: Dict[str, Any]) -> str:
        """Analyze and explain ML model results."""
        self._check_api_availability()
        
        prompt = f"""
        As a machine learning expert, analyze these model results and provide insights:

        ML Results:
        {json.dumps(ml_results, indent=2, default=str)}

        Please provide:
        1. Model performance interpretation
        2. Best model recommendation and reasoning
        3. Feature importance insights (if available)
        4. Model limitations and considerations
        5. Recommendations for model improvement
        6. Business implications of the results

        Make the explanation accessible to both technical and non-technical stakeholders.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a machine learning expert explaining model results."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing ML results: {str(e)}"
    
    def answer_custom_question(self, df: pd.DataFrame, question: str) -> str:
        """Answer custom questions about the dataset."""
        self._check_api_availability()
        
        # Prepare dataset context
        context = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': dict(df.dtypes.astype(str)),
            'sample_data': df.head(5).to_dict('records'),
            'summary_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        prompt = f"""
        Dataset Context:
        {json.dumps(context, indent=2, default=str)}

        User Question: {question}

        As a data analyst, answer the user's question about this dataset. Provide specific insights based on the actual data shown above. If the question cannot be fully answered with the available information, explain what additional data or analysis would be needed.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data analyst answering questions about datasets."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error answering question: {str(e)}"
