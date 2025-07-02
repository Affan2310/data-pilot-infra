import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional

class ReportGenerator:
    """Generate comprehensive analysis reports in various formats."""
    
    def __init__(self):
        self.report_timestamp = datetime.now().isoformat()
    
    def generate_profile_report(self, df: pd.DataFrame, profile_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive data profiling report."""
        report = {
            'report_type': 'Data Profile Report',
            'generated_at': self.report_timestamp,
            'dataset_info': {
                'name': 'User Dataset',
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'data_profile': profile_results,
            'data_sample': df.head(10).to_dict('records'),
            'summary': self._generate_profile_summary(df, profile_results)
        }
        
        return report
    
    def generate_ml_report(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate machine learning results report."""
        report = {
            'report_type': 'ML Model Results Report',
            'generated_at': self.report_timestamp,
            'model_results': ml_results,
            'summary': self._generate_ml_summary(ml_results)
        }
        
        return report
    
    def generate_complete_report(self, df: pd.DataFrame, 
                               profile_results: Optional[Dict[str, Any]] = None,
                               ml_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis report combining all components."""
        report = {
            'report_type': 'Complete Data Analysis Report',
            'generated_at': self.report_timestamp,
            'dataset_overview': {
                'shape': df.shape,
                'columns': list(df.columns),
                'data_types': dict(df.dtypes.astype(str)),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'missing_values_total': int(df.isnull().sum().sum()),
                'duplicate_rows': int(df.duplicated().sum())
            },
            'data_sample': df.head(5).to_dict('records'),
            'executive_summary': self._generate_executive_summary(df, profile_results, ml_results)
        }
        
        # Add profiling results if available
        if profile_results:
            report['data_profiling'] = profile_results
            report['data_quality_assessment'] = self._generate_quality_assessment(profile_results)
        
        # Add ML results if available
        if ml_results:
            report['machine_learning'] = ml_results
            report['model_recommendations'] = self._generate_model_recommendations(ml_results)
        
        # Add technical appendix
        report['technical_appendix'] = self._generate_technical_appendix(df)
        
        return report
    
    def _generate_profile_summary(self, df: pd.DataFrame, profile_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for data profiling report."""
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        summary = {
            'total_columns': df.shape[1],
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'total_missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': float(profile_results.get('missing_percentage', 0)),
            'duplicate_rows': int(profile_results.get('duplicates', 0)),
            'data_quality_score': self._calculate_quality_score(profile_results),
            'key_findings': self._extract_key_findings(profile_results)
        }
        
        return summary
    
    def _generate_ml_summary(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for ML results report."""
        if not ml_results.get('model_performance'):
            return {'error': 'No model performance data available'}
        
        performance_data = ml_results['model_performance']
        problem_type = ml_results.get('problem_type', 'unknown')
        
        # Find best model based on problem type
        if problem_type == 'classification':
            best_model = max(performance_data, key=lambda x: x.get('Accuracy', 0))
            metric_name = 'Accuracy'
        else:
            best_model = max(performance_data, key=lambda x: x.get('R² Score', -float('inf')))
            metric_name = 'R² Score'
        
        summary = {
            'problem_type': problem_type,
            'models_evaluated': len(performance_data),
            'best_model': {
                'name': best_model.get('Model', 'Unknown'),
                'performance': best_model.get(metric_name, 0),
                'metric': metric_name
            },
            'feature_count': len(ml_results.get('feature_columns', [])),
            'target_variable': ml_results.get('target_column', 'Unknown'),
            'test_samples': ml_results.get('test_size', 0),
            'model_comparison': performance_data
        }
        
        return summary
    
    def _generate_executive_summary(self, df: pd.DataFrame, 
                                  profile_results: Optional[Dict[str, Any]] = None,
                                  ml_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate executive summary for complete report."""
        summary = {
            'dataset_overview': f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns with {df.isnull().sum().sum()} missing values.",
            'data_quality': "Data quality assessment completed." if profile_results else "Data quality assessment not performed.",
            'machine_learning': "Machine learning models built and evaluated." if ml_results else "Machine learning analysis not performed.",
            'recommendations': []
        }
        
        # Add data quality recommendations
        if profile_results and profile_results.get('cleaning_suggestions'):
            summary['recommendations'].extend(profile_results['cleaning_suggestions'][:3])
        
        # Add ML recommendations
        if ml_results and ml_results.get('model_performance'):
            best_model = max(ml_results['model_performance'], 
                           key=lambda x: x.get('Accuracy', x.get('R² Score', 0)))
            summary['recommendations'].append(f"Consider using {best_model.get('Model', 'the best performing model')} for production deployment.")
        
        if not summary['recommendations']:
            summary['recommendations'] = ["Perform comprehensive data analysis to identify improvement opportunities."]
        
        return summary
    
    def _generate_quality_assessment(self, profile_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data quality assessment."""
        quality_score = self._calculate_quality_score(profile_results)
        
        assessment = {
            'overall_score': quality_score,
            'score_interpretation': self._interpret_quality_score(quality_score),
            'issues_identified': profile_results.get('quality_issues', {}),
            'cleaning_priority': self._prioritize_cleaning_tasks(profile_results),
            'recommendations': profile_results.get('cleaning_suggestions', [])
        }
        
        return assessment
    
    def _generate_model_recommendations(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model deployment recommendations."""
        if not ml_results.get('model_performance'):
            return {'error': 'No model performance data available'}
        
        performance_data = ml_results['model_performance']
        problem_type = ml_results.get('problem_type', 'unknown')
        
        # Sort models by performance
        if problem_type == 'classification':
            sorted_models = sorted(performance_data, key=lambda x: x.get('Accuracy', 0), reverse=True)
            primary_metric = 'Accuracy'
        else:
            sorted_models = sorted(performance_data, key=lambda x: x.get('R² Score', -float('inf')), reverse=True)
            primary_metric = 'R² Score'
        
        recommendations = {
            'best_model': {
                'name': sorted_models[0].get('Model', 'Unknown'),
                'performance': sorted_models[0].get(primary_metric, 0),
                'reason': f"Highest {primary_metric} score"
            },
            'alternative_models': [
                {
                    'name': model.get('Model', 'Unknown'),
                    'performance': model.get(primary_metric, 0)
                } for model in sorted_models[1:3]
            ],
            'deployment_considerations': [
                "Validate model performance on additional test data",
                "Consider model interpretability requirements",
                "Evaluate computational requirements for deployment",
                "Implement monitoring for model drift"
            ],
            'next_steps': [
                "Hyperparameter tuning for best performing model",
                "Feature engineering exploration",
                "Cross-validation with different data splits",
                "Model ensemble consideration"
            ]
        }
        
        return recommendations
    
    def _generate_technical_appendix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate technical appendix with detailed information."""
        appendix = {
            'column_details': {},
            'data_types_breakdown': dict(df.dtypes.value_counts().astype(str)),
            'missing_values_by_column': dict(df.isnull().sum()),
            'unique_values_by_column': dict(df.nunique()),
            'memory_usage_by_column': dict(df.memory_usage(deep=True))
        }
        
        # Detailed column analysis
        for col in df.columns:
            col_info = {
                'data_type': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'unique_values': int(df[col].nunique()),
                'memory_usage_bytes': int(df[col].memory_usage(deep=True))
            }
            
            # Add statistics for numeric columns
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'median': float(df[col].median()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None
                })
            
            # Add info for categorical columns
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                col_info.update({
                    'most_frequent_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'cardinality_ratio': float(df[col].nunique() / len(df))
                })
            
            appendix['column_details'][col] = col_info
        
        return appendix
    
    def _calculate_quality_score(self, profile_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Deduct for missing values
        missing_penalty = min(profile_results.get('missing_percentage', 0) * 2, 30)
        score -= missing_penalty
        
        # Deduct for quality issues
        quality_issues = profile_results.get('quality_issues', {})
        issue_count = sum(len(issues) for issues in quality_issues.values())
        score -= min(issue_count * 5, 20)
        
        # Deduct for duplicates (if significant)
        duplicates = profile_results.get('duplicates', 0)
        if duplicates > 0:
            total_rows = profile_results.get('shape', [0])[0]
            if total_rows > 0:
                dup_penalty = min((duplicates / total_rows) * 100 * 0.5, 15)
                score -= dup_penalty
        
        return max(score, 0)
    
    def _interpret_quality_score(self, score: float) -> str:
        """Interpret data quality score."""
        if score >= 90:
            return "Excellent - High quality data with minimal issues"
        elif score >= 75:
            return "Good - Data quality is acceptable with minor issues"
        elif score >= 60:
            return "Fair - Some data quality issues that should be addressed"
        elif score >= 40:
            return "Poor - Significant data quality issues requiring attention"
        else:
            return "Critical - Major data quality problems that must be resolved"
    
    def _prioritize_cleaning_tasks(self, profile_results: Dict[str, Any]) -> list:
        """Prioritize data cleaning tasks by importance."""
        tasks = []
        
        # High priority: Missing values
        if profile_results.get('missing_percentage', 0) > 10:
            tasks.append("HIGH: Address missing values (>10% of data)")
        
        # High priority: Duplicates
        if profile_results.get('duplicates', 0) > 0:
            tasks.append("HIGH: Remove duplicate records")
        
        # Medium priority: Quality issues
        quality_issues = profile_results.get('quality_issues', {})
        if quality_issues.get('high_cardinality'):
            tasks.append("MEDIUM: Review high cardinality categorical variables")
        
        if quality_issues.get('potential_outliers'):
            tasks.append("MEDIUM: Investigate potential outliers")
        
        # Low priority: Optimization
        if profile_results.get('memory_usage', 0) > 100_000_000:  # > 100MB
            tasks.append("LOW: Optimize memory usage and data types")
        
        return tasks if tasks else ["No critical cleaning tasks identified"]
    
    def _extract_key_findings(self, profile_results: Dict[str, Any]) -> list:
        """Extract key findings from profile results."""
        findings = []
        
        # Missing data findings
        missing_pct = profile_results.get('missing_percentage', 0)
        if missing_pct > 20:
            findings.append(f"High missing data rate: {missing_pct:.1f}%")
        elif missing_pct > 5:
            findings.append(f"Moderate missing data: {missing_pct:.1f}%")
        
        # Quality issues
        quality_issues = profile_results.get('quality_issues', {})
        if quality_issues.get('high_cardinality'):
            findings.append(f"High cardinality columns detected: {len(quality_issues['high_cardinality'])}")
        
        if quality_issues.get('potential_outliers'):
            findings.append(f"Columns with outliers: {len(quality_issues['potential_outliers'])}")
        
        # Data distribution
        numeric_summary = profile_results.get('numeric_summary', {})
        if numeric_summary:
            skewed_cols = [col for col, stats in numeric_summary.items() 
                          if stats.get('skewness', 0) and abs(stats['skewness']) > 1]
            if skewed_cols:
                findings.append(f"Highly skewed distributions: {len(skewed_cols)} columns")
        
        return findings if findings else ["No significant issues identified"]
