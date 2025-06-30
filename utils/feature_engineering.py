import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced automated feature engineering for machine learning."""
    
    def __init__(self):
        self.feature_transformations = {}
        self.generated_features = {}
        self.feature_importance_scores = {}
        
    def auto_engineer_features(self, df: pd.DataFrame, target_col: str, 
                             max_features: int = 50) -> Dict[str, Any]:
        """
        Automatically engineer features using various techniques.
        """
        try:
            results = {
                'original_features': len(df.columns) - 1,
                'engineered_features': {},
                'feature_importance': {},
                'recommendations': [],
                'transformation_summary': {}
            }
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Determine problem type
            is_classification = self._is_classification_problem(y)
            
            # 1. Basic feature engineering
            X_engineered = X.copy()
            
            # 2. Handle missing values intelligently
            X_engineered = self._smart_missing_value_imputation(X_engineered)
            
            # 3. Generate interaction features
            interaction_features = self._create_interaction_features(X_engineered, max_interactions=10)
            results['engineered_features']['interactions'] = len(interaction_features.columns) - len(X_engineered.columns)
            
            # 4. Generate polynomial features for numeric columns
            poly_features = self._create_polynomial_features(X_engineered, degree=2, max_features=15)
            results['engineered_features']['polynomial'] = len(poly_features.columns) - len(interaction_features.columns)
            
            # 5. Generate statistical features
            stat_features = self._create_statistical_features(poly_features)
            results['engineered_features']['statistical'] = len(stat_features.columns) - len(poly_features.columns)
            
            # 6. Generate time-based features if date columns exist
            time_features = self._create_time_features(stat_features)
            results['engineered_features']['temporal'] = len(time_features.columns) - len(stat_features.columns)
            
            # 7. Generate categorical encoding features
            encoded_features = self._advanced_categorical_encoding(time_features, y, is_classification)
            results['engineered_features']['categorical'] = len(encoded_features.columns) - len(time_features.columns)
            
            # 8. Feature selection to keep best features
            selected_features, importance_scores = self._intelligent_feature_selection(
                encoded_features, y, is_classification, max_features
            )
            
            results['feature_importance'] = importance_scores
            results['final_features'] = len(selected_features.columns)
            
            # 9. Generate dimensionality reduction features
            pca_features = self._create_pca_features(selected_features, n_components=min(10, len(selected_features.columns)//2))
            results['engineered_features']['pca'] = len(pca_features.columns) - len(selected_features.columns)
            
            # 10. Generate recommendations
            recommendations = self._generate_feature_recommendations(results, df, is_classification)
            results['recommendations'] = recommendations
            
            # Store transformation summary
            results['transformation_summary'] = {
                'total_original_features': len(X.columns),
                'total_generated_features': len(pca_features.columns),
                'feature_expansion_ratio': len(pca_features.columns) / len(X.columns),
                'recommended_for_modeling': True if len(pca_features.columns) > len(X.columns) else False
            }
            
            # Return final engineered dataset
            final_df = pca_features.copy()
            final_df[target_col] = y
            
            results['engineered_dataset'] = final_df
            
            return results
            
        except Exception as e:
            return {
                'error': f"Feature engineering failed: {str(e)}",
                'original_features': len(df.columns) - 1,
                'engineered_features': {},
                'recommendations': ["Feature engineering encountered errors - review data quality"]
            }
    
    def _is_classification_problem(self, y: pd.Series) -> bool:
        """Determine if this is a classification problem."""
        return y.dtype == 'object' or y.nunique() <= 20
    
    def _smart_missing_value_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value imputation based on feature types."""
        df_imputed = df.copy()
        
        for col in df_imputed.columns:
            if df_imputed[col].isnull().sum() > 0:
                if df_imputed[col].dtype in ['object', 'category']:
                    # For categorical: use mode or create 'missing' category
                    if df_imputed[col].mode().empty:
                        df_imputed[col] = df_imputed[col].fillna('missing')
                    else:
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
                else:
                    # For numeric: use median for skewed data, mean for normal
                    if abs(df_imputed[col].skew()) > 1:
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                    else:
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
        
        return df_imputed
    
    def _create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        df_interactions = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        interactions_created = 0
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if interactions_created >= max_interactions:
                    break
                
                # Multiplicative interaction
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]
                
                # Ratio interaction (avoid division by zero)
                ratio_name = f"{col1}_div_{col2}"
                df_interactions[ratio_name] = df[col1] / (df[col2] + 1e-8)
                
                interactions_created += 2
                
                if interactions_created >= max_interactions:
                    break
        
        return df_interactions
    
    def _create_polynomial_features(self, df: pd.DataFrame, degree: int = 2, max_features: int = 15) -> pd.DataFrame:
        """Create polynomial features for numeric columns."""
        df_poly = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to prevent explosion of features
        selected_cols = numeric_cols[:min(5, len(numeric_cols))]
        
        if len(selected_cols) > 0:
            try:
                poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
                X_numeric = df[selected_cols]
                X_poly = poly.fit_transform(X_numeric)
                
                # Create feature names
                feature_names = poly.get_feature_names_out(selected_cols)
                
                # Add only new polynomial features (not the original ones)
                for i, name in enumerate(feature_names):
                    if name not in selected_cols and len(df_poly.columns) < len(df.columns) + max_features:
                        df_poly[f"poly_{name}"] = X_poly[:, i]
            
            except Exception:
                pass  # Skip if polynomial features fail
        
        return df_poly
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from numeric columns."""
        df_stats = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            numeric_data = df[numeric_cols]
            
            # Row-wise statistics
            df_stats['row_mean'] = numeric_data.mean(axis=1)
            df_stats['row_std'] = numeric_data.std(axis=1)
            df_stats['row_min'] = numeric_data.min(axis=1)
            df_stats['row_max'] = numeric_data.max(axis=1)
            df_stats['row_range'] = df_stats['row_max'] - df_stats['row_min']
            
            # Percentile features
            df_stats['row_25pct'] = numeric_data.quantile(0.25, axis=1)
            df_stats['row_75pct'] = numeric_data.quantile(0.75, axis=1)
            
            # Count-based features
            df_stats['count_positive'] = (numeric_data > 0).sum(axis=1)
            df_stats['count_negative'] = (numeric_data < 0).sum(axis=1)
            df_stats['count_zero'] = (numeric_data == 0).sum(axis=1)
        
        return df_stats
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime columns."""
        df_time = df.copy()
        
        for col in df.columns:
            # Try to identify datetime columns
            if df[col].dtype == 'object':
                try:
                    # Attempt to parse as datetime
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                    if dt_series.notna().sum() > len(df) * 0.5:  # If more than 50% parseable
                        # Extract time features
                        df_time[f"{col}_year"] = dt_series.dt.year
                        df_time[f"{col}_month"] = dt_series.dt.month
                        df_time[f"{col}_day"] = dt_series.dt.day
                        df_time[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                        df_time[f"{col}_quarter"] = dt_series.dt.quarter
                        df_time[f"{col}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
                        
                        # Remove original datetime column
                        df_time = df_time.drop(columns=[col])
                except:
                    continue
        
        return df_time
    
    def _advanced_categorical_encoding(self, df: pd.DataFrame, y: pd.Series, is_classification: bool) -> pd.DataFrame:
        """Advanced categorical encoding techniques."""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            try:
                # Target encoding for high-cardinality categorical variables
                if df[col].nunique() > 10:
                    if is_classification:
                        # Mean target encoding for classification
                        target_mean = y.groupby(df[col]).apply(lambda x: (x == x.mode()[0]).mean() if not x.mode().empty else 0)
                    else:
                        # Mean target encoding for regression
                        target_mean = y.groupby(df[col]).mean()
                    
                    df_encoded[f"{col}_target_encoded"] = df[col].map(target_mean).fillna(y.mean())
                
                # Frequency encoding
                freq_map = df[col].value_counts().to_dict()
                df_encoded[f"{col}_frequency"] = df[col].map(freq_map)
                
                # One-hot encoding for low-cardinality variables
                if df[col].nunique() <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Remove original categorical column
                df_encoded = df_encoded.drop(columns=[col])
                
            except Exception:
                # If encoding fails, just drop the column
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def _intelligent_feature_selection(self, df: pd.DataFrame, y: pd.Series, 
                                     is_classification: bool, max_features: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Intelligent feature selection using multiple methods."""
        # Ensure all features are numeric
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return df, {}
        
        # Handle any remaining NaN values
        numeric_df = numeric_df.fillna(0)
        
        try:
            # Choose appropriate scoring function
            if is_classification:
                score_func = f_classif
                mutual_info_func = mutual_info_classif
            else:
                score_func = f_regression
                mutual_info_func = mutual_info_regression
            
            # Method 1: Statistical tests
            selector_stats = SelectKBest(score_func=score_func, k=min(max_features, len(numeric_df.columns)))
            X_stats = selector_stats.fit_transform(numeric_df, y)
            selected_features_stats = numeric_df.columns[selector_stats.get_support()]
            
            # Method 2: Mutual information
            try:
                selector_mi = SelectKBest(score_func=mutual_info_func, k=min(max_features, len(numeric_df.columns)))
                X_mi = selector_mi.fit_transform(numeric_df, y)
                selected_features_mi = numeric_df.columns[selector_mi.get_support()]
                
                # Combine both methods
                combined_features = list(set(selected_features_stats) | set(selected_features_mi))
            except:
                combined_features = list(selected_features_stats)
            
            # Limit final features
            final_features = combined_features[:max_features]
            
            # Calculate importance scores
            importance_scores = {}
            if len(final_features) > 0:
                try:
                    stats_scores = selector_stats.scores_
                    for i, feature in enumerate(selected_features_stats):
                        if feature in final_features:
                            importance_scores[feature] = float(stats_scores[i])
                except:
                    # Default equal importance if scoring fails
                    for feature in final_features:
                        importance_scores[feature] = 1.0
            
            selected_df = numeric_df[final_features] if final_features else numeric_df.iloc[:, :max_features]
            
            return selected_df, importance_scores
            
        except Exception:
            # Fallback: just return top features by variance
            feature_vars = numeric_df.var().sort_values(ascending=False)
            top_features = feature_vars.head(max_features).index.tolist()
            importance_scores = {f: float(feature_vars[f]) for f in top_features}
            
            return numeric_df[top_features], importance_scores
    
    def _create_pca_features(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Create PCA features for dimensionality reduction."""
        if len(df.columns) < 2 or n_components <= 0:
            return df
        
        try:
            # Standardize features before PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df.fillna(0))
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, len(df.columns), len(df)))
            X_pca = pca.fit_transform(X_scaled)
            
            # Create PCA feature dataframe
            pca_df = df.copy()
            for i in range(X_pca.shape[1]):
                pca_df[f'pca_component_{i+1}'] = X_pca[:, i]
            
            return pca_df
            
        except Exception:
            return df
    
    def _generate_feature_recommendations(self, results: Dict[str, Any], 
                                        original_df: pd.DataFrame, is_classification: bool) -> List[str]:
        """Generate recommendations for feature engineering."""
        recommendations = []
        
        # Check feature expansion
        expansion_ratio = results['transformation_summary'].get('feature_expansion_ratio', 1)
        if expansion_ratio > 2:
            recommendations.append(f"Successfully expanded features by {expansion_ratio:.1f}x - good feature diversity created")
        elif expansion_ratio < 1.5:
            recommendations.append("Limited feature expansion - consider domain-specific feature creation")
        
        # Check for specific feature types
        if results['engineered_features'].get('interactions', 0) > 0:
            recommendations.append("Interaction features created - may capture non-linear relationships")
        
        if results['engineered_features'].get('polynomial', 0) > 0:
            recommendations.append("Polynomial features added - useful for capturing curve relationships")
        
        if results['engineered_features'].get('temporal', 0) > 0:
            recommendations.append("Time-based features detected and engineered")
        
        # Data quality recommendations
        missing_percentage = (original_df.isnull().sum().sum() / (len(original_df) * len(original_df.columns))) * 100
        if missing_percentage > 10:
            recommendations.append(f"High missing data ({missing_percentage:.1f}%) - consider data collection improvements")
        
        # Model-specific recommendations
        if is_classification:
            recommendations.append("Classification problem detected - engineered features for categorical prediction")
        else:
            recommendations.append("Regression problem detected - engineered features for numerical prediction")
        
        if not recommendations:
            recommendations.append("Feature engineering completed successfully")
        
        return recommendations
    
    def get_feature_importance_summary(self, importance_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of feature importance."""
        if not importance_scores:
            return {}
        
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_5_features': sorted_features[:5],
            'total_features': len(sorted_features),
            'importance_range': {
                'max': max(importance_scores.values()),
                'min': min(importance_scores.values()),
                'mean': np.mean(list(importance_scores.values()))
            }
        }