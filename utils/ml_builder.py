import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class MLBuilder:
    """Automated machine learning model builder and evaluator."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.preprocessor = None
        
    def build_models(self, df: pd.DataFrame, target_col: str, 
                    feature_cols: list, problem_type: str) -> dict:
        """
        Build and evaluate multiple ML models automatically.
        
        Args:
            df: Input dataframe
            target_col: Target variable column name
            feature_cols: List of feature column names
            problem_type: 'classification' or 'regression'
            
        Returns:
            Dictionary containing model performance results
        """
        try:
            # Prepare data
            X, y = self._prepare_data(df, target_col, feature_cols, problem_type)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None
            )
            
            # Get models based on problem type
            models = self._get_models(problem_type)
            
            # Train and evaluate models
            results = []
            feature_importance = {}
            
            for name, model in models.items():
                try:
                    # Create preprocessing pipeline
                    pipeline = self._create_pipeline(X, model)
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_test, y_pred, problem_type)
                    metrics['Model'] = name
                    results.append(metrics)
                    
                    # Extract feature importance if available
                    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                        importance = pipeline.named_steps['model'].feature_importances_
                        feature_names = self._get_feature_names(X)
                        feature_importance[name] = dict(zip(feature_names, importance))
                    
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
            
            # Prepare return results
            return_results = {
                'model_performance': results,
                'problem_type': problem_type,
                'target_column': target_col,
                'feature_columns': feature_cols,
                'data_shape': df.shape,
                'test_size': len(X_test)
            }
            
            if feature_importance:
                return_results['feature_importance'] = feature_importance[list(feature_importance.keys())[0]]
            
            return return_results
            
        except Exception as e:
            raise Exception(f"Error in model building: {str(e)}")
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str, 
                     feature_cols: list, problem_type: str) -> tuple:
        """Prepare data for model training."""
        # Select features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values in target
        if y.isnull().sum() > 0:
            X = X[~y.isnull()]
            y = y.dropna()
        
        # Encode target variable for classification
        if problem_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y
    
    def _create_pipeline(self, X: pd.DataFrame, model) -> Pipeline:
        """Create preprocessing and modeling pipeline."""
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessors
        preprocessors = []
        
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            preprocessors.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessors.append(('cat', categorical_transformer, categorical_features))
        
        # Create column transformer
        if preprocessors:
            preprocessor = ColumnTransformer(transformers=preprocessors)
        else:
            preprocessor = 'passthrough'
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        return pipeline
    
    def _get_models(self, problem_type: str) -> dict:
        """Get appropriate models based on problem type."""
        if problem_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
            }
        else:  # regression
            return {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVR': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
            }
    
    def _calculate_metrics(self, y_true, y_pred, problem_type: str) -> dict:
        """Calculate appropriate metrics based on problem type."""
        if problem_type == 'classification':
            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['ROC AUC'] = roc_auc_score(y_true, y_pred)
                except:
                    metrics['ROC AUC'] = 0.0
        
        else:  # regression
            metrics = {
                'R² Score': r2_score(y_true, y_pred),
                'Mean Squared Error': mean_squared_error(y_true, y_pred),
                'Root Mean Squared Error': np.sqrt(mean_squared_error(y_true, y_pred)),
                'Mean Absolute Error': mean_absolute_error(y_true, y_pred)
            }
        
        return metrics
    
    def _get_feature_names(self, X: pd.DataFrame) -> list:
        """Get feature names after preprocessing."""
        return X.columns.tolist()
    
    def predict_with_best_model(self, df: pd.DataFrame, target_col: str, 
                              feature_cols: list, problem_type: str, new_data: pd.DataFrame):
        """Make predictions using the best performing model."""
        # This would be implemented for making predictions on new data
        # For now, returning placeholder
        return {"message": "Prediction functionality to be implemented"}
    
    def get_model_interpretation(self, model_name: str) -> dict:
        """Get model interpretation and feature importance."""
        # This would provide detailed model interpretation
        # For now, returning placeholder
        return {"message": "Model interpretation functionality to be implemented"}
