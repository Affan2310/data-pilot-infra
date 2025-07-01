import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataProfiler:
    """Handles comprehensive data profiling and quality assessment."""
    
    def __init__(self):
        self.profile_results = {}
    
    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data profile including statistics, 
        data types, missing values, and quality issues.
        """
        profile = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'shape': df.shape,
            'dtypes': dict(df.dtypes.astype(str)),
            'missing_values': dict(df.isnull().sum()),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': self._get_numeric_summary(df),
            'categorical_summary': self._get_categorical_summary(df),
            'quality_issues': self._identify_quality_issues(df),
            'cleaning_suggestions': self._generate_cleaning_suggestions(df)
        }
        
        self.profile_results = profile
        return profile
    
    def _get_numeric_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_cols:
            try:
                summary[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'zeros': (df[col] == 0).sum(),
                    'outliers': self._count_outliers(df[col])
                }
            except Exception as e:
                summary[col] = {'error': str(e)}
        
        return summary
    
    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        summary = {}
        
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()
                summary[col] = {
                    'unique_values': df[col].nunique(),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                    'cardinality_ratio': df[col].nunique() / len(df)
                }
            except Exception as e:
                summary[col] = {'error': str(e)}
        
        return summary
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series < lower_bound) | (series > upper_bound)).sum()
        except:
            return 0
    
    def _identify_quality_issues(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify potential data quality issues."""
        issues = {
            'high_cardinality': [],
            'potential_outliers': [],
            'high_missing': [],
            'constant_columns': [],
            'duplicate_columns': []
        }
        
        # High cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].nunique() / len(df) > 0.9:
                issues['high_cardinality'].append(col)
        
        # Columns with many outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outlier_count = self._count_outliers(df[col])
            if outlier_count > len(df) * 0.05:  # More than 5% outliers
                issues['potential_outliers'].append(col)
        
        # High missing value columns
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0.5:  # More than 50% missing
                issues['high_missing'].append(col)
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                issues['constant_columns'].append(col)
        
        # Duplicate columns
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    issues['duplicate_columns'].append(f"{col1} = {col2}")
        
        return issues
    
    def _generate_cleaning_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate data cleaning suggestions based on profile analysis."""
        suggestions = []
        
        # Missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            suggestions.append(f"Handle missing values in columns: {', '.join(missing_cols)}")
        
        # Duplicates
        if df.duplicated().sum() > 0:
            suggestions.append(f"Remove {df.duplicated().sum()} duplicate rows")
        
        # Data type optimization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype == 'float64' and df[col].nunique() < 1000:
                suggestions.append(f"Consider converting '{col}' to categorical or integer type")
        
        # Outlier handling
        for col in numeric_cols:
            outlier_count = self._count_outliers(df[col])
            if outlier_count > 0:
                suggestions.append(f"Review {outlier_count} outliers in column '{col}'")
        
        # Memory optimization
        if df.memory_usage(deep=True).sum() > 100_000_000:  # > 100MB
            suggestions.append("Consider memory optimization for large dataset")
        
        return suggestions if suggestions else ["Data appears to be in good quality!"]
    
    def generate_human_friendly_report(self, profile: Dict[str, Any]) -> str:
        """Create a human-friendly summary text for data profiling results."""
        try:
            lines = []
            lines.append("📄 Data Profiling Summary")
            lines.append(f"📅 Timestamp: {profile.get('timestamp')}")
            lines.append(f"📊 Shape: {profile.get('shape')[0]} rows × {profile.get('shape')[1]} columns")
            lines.append(f"💾 Memory usage: {profile.get('memory_usage')} bytes")
            lines.append(f"🔍 Missing values: {profile.get('missing_percentage'):.2f}%")
            lines.append(f"📑 Duplicate rows: {profile.get('duplicates')}")
            lines.append("")
            
            # Data types
            lines.append("📂 Column Data Types:")
            for col, dtype in profile.get('dtypes', {}).items():
                lines.append(f"  - {col}: {dtype}")
            lines.append("")
            
            # Numeric summary
            lines.append("📈 Numeric Columns Summary:")
            for col, stats in profile.get('numeric_summary', {}).items():
                lines.append(f"  • {col}: Mean {stats.get('mean'):.2f}, Std {stats.get('std'):.2f}, Min {stats.get('min')}, Max {stats.get('max')}")
            lines.append("")
            
            # Categorical summary
            lines.append("🏷️ Categorical Columns Summary:")
            for col, stats in profile.get('categorical_summary', {}).items():
                lines.append(f"  • {col}: {stats.get('unique_values')} unique values (Most frequent: {stats.get('most_frequent')})")
            lines.append("")
            
            # Quality issues
            issues = profile.get('quality_issues', {})
            lines.append("⚠️ Quality Issues:")
            for issue, cols in issues.items():
                if cols:
                    lines.append(f"  - {issue.replace('_', ' ').capitalize()}: {', '.join(cols)}")
            if not any(issues.values()):
                lines.append("  - No significant issues detected.")
            lines.append("")
            
            # Cleaning suggestions
            lines.append("🛠 Cleaning Suggestions:")
            for suggestion in profile.get('cleaning_suggestions', []):
                lines.append(f"✅ {suggestion}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error generating human-friendly report: {str(e)}"

