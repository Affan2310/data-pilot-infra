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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
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
        
    def build_models(self, df, target_col, features=[], problem_type='classification'):
        try:
            X = df[features] if features else df.drop(columns=[target_col])
            y = df[target_col]

            # Handle stratify for classification safely
            stratify = None
            if problem_type == 'classification':
                class_counts = y.value_counts()
                if class_counts.min() >= 2:
                    stratify = y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )

            models = []
            if problem_type == 'classification':
                models = [
                    ('Logistic Regression', LogisticRegression(max_iter=1000)),
                    ('Random Forest Classifier', RandomForestClassifier()),
                    ('SVM Classifier', SVC()),
                    ('XGBoost Classifier', XGBClassifier(eval_metric='logloss', use_label_encoder=False))
                ]
                metric_name = 'accuracy'
            else:
                models = [
                    ('Linear Regression', LinearRegression()),
                    ('Random Forest Regressor', RandomForestRegressor()),
                    ('SVR', SVR()),
                    ('XGBoost Regressor', XGBRegressor())
                ]
                metric_name = 'rmse'

            results = []
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
                else:
                    score = mean_squared_error(y_test, y_pred, squared=False)
                    cv_score = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()

                importance = None
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    coef = model.coef_
                    if coef.ndim > 1:
                        coef = coef[0]
                    importance = dict(zip(X.columns, coef))

                results.append({
                    'model_name': name,
                    'test_score': score,
                    'cv_score': cv_score,
                    'feature_importance': importance
                })

            best_model = max(results, key=lambda x: x['cv_score']) if problem_type == 'classification' \
                         else min(results, key=lambda x: x['cv_score'])

            return {
                'best_model': best_model,
                'all_results': results
            }

        except Exception as e:
            return {'error': f"Model building failed: {str(e)}"}
    
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
