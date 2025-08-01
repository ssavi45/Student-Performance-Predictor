import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             VotingRegressor, StackingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class ImprovedStudentPerformancePredictor:
    def __init__(self):
        self.preprocessor = None
        self.models = {}
        self.best_model = None
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Enhanced data loading and preprocessing"""
        df = pd.read_csv('study.csv')
        
        # Drop the date column
        df = df.drop('29/04/2023', axis=1)
        
        # Rename columns to simpler names
        column_mapping = {
            "gender\nstudent's sex (binary: 'F' - female or 'M' - male)": 'gender',
            "age\nstudent's age (numeric: from 15 to 22)": 'age',
            "adress\nstudent's home address type (binary: 'U' - urban or 'R' - rural)": 'address',
            "famsize\nfamily size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)": 'family_size',
            "Pstatus\nparent's cohabitation status (binary: 'T' - living together or 'A' - apart)": 'parent_status',
            'M_Education': 'm_education',
            'F_education': 'f_education',
            "Mjob\nmother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or other": 'm_job',
            "Fjob\nfather's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or other": 'f_job',
            'avarage_tution_fee_cost': 'average_tuition_fee_cost'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert time_spent_with_friends to numeric
        df['time_spent_with_friends'] = pd.to_numeric(df['time_spent_with_friends'], errors='coerce')
        
        # Clean text data
        text_columns = ['parent_status', 'smoker', 'relationship_breakdown', 'm_job', 'f_job']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.capitalize()
        
        # Handle gender encoding
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
        
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        return df
    
    def create_features(self, df):
        """Advanced feature engineering"""
        df_features = df.copy()
        
        # 1. Interaction features
        df_features['ssc_age_interaction'] = df_features['ssc_result'] * df_features['age']
        df_features['education_avg'] = (df_features['m_education'] + df_features['f_education']) / 2
        df_features['tuition_per_friend_time'] = df_features['average_tuition_fee_cost'] / (df_features['time_spent_with_friends'] + 1)
        
        # 2. Polynomial features for key continuous variables
        df_features['ssc_squared'] = df_features['ssc_result'] ** 2
        df_features['age_squared'] = df_features['age'] ** 2
        
        # 3. Binning continuous variables
        df_features['age_group'] = pd.cut(df_features['age'], bins=[0, 18, 20, 25], labels=['young', 'medium', 'old'])
        df_features['ssc_performance'] = pd.cut(df_features['ssc_result'], bins=[0, 3.5, 4.5, 5.0], labels=['low', 'medium', 'high'])
        
        # 4. Family stability indicator
        df_features['family_stability'] = ((df_features['parent_status'] == 'T') & 
                                         (df_features['relationship_breakdown'] == 'No')).astype(int)
        
        # 5. Educational support score
        df_features['educational_support'] = df_features['m_education'] + df_features['f_education']
        
        return df_features
    
    def create_preprocessor(self, X):
        """Create comprehensive preprocessing pipeline"""
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target if accidentally included
        if 'hsc_result' in numeric_features:
            numeric_features.remove('hsc_result')
        if 'hsc_result' in categorical_features:
            categorical_features.remove('hsc_result')
        
        # Numeric pipeline: impute with median, then scale
        numeric_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute with mode, then one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        return preprocessor, numeric_features, categorical_features
    
    def create_models(self):
        """Create diverse set of models with different approaches"""
        models = {
            # Tree-based models
            'RandomForest': RandomForestRegressor(random_state=42),
            'ExtraTrees': ExtraTreesRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42, eval_metric='rmse'),
            'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
            
            # Linear models
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            
            # Other algorithms
            'SVR': SVR(),
            'MLPRegressor': MLPRegressor(random_state=42, max_iter=500)
        }
        
        return models
    
    def get_hyperparameter_grids(self):
        """Define hyperparameter grids for tuning"""
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'SVR': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
        return param_grids
    
    def train_and_evaluate_models(self, X, y):
        """Train models with hyperparameter tuning and cross-validation"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit preprocessor on training data
        self.preprocessor.fit(X_train)
        X_train_processed = self.preprocessor.transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print("Training and evaluating models...")
        print("=" * 60)
        
        models = self.create_models()
        param_grids = self.get_hyperparameter_grids()
        results = {}
        
        for name, model in models.items():
            print(f"\n🔍 Training {name}...")
            
            try:
                # Hyperparameter tuning for selected models
                if name in param_grids:
                    grid_model = GridSearchCV(
                        model, param_grids[name], 
                        cv=5, scoring='r2', n_jobs=-1, verbose=0
                    )
                    grid_model.fit(X_train_processed, y_train)
                    best_model = grid_model.best_estimator_
                    print(f"   Best params: {grid_model.best_params_}")
                else:
                    # Use default parameters
                    best_model = model
                    best_model.fit(X_train_processed, y_train)
                
                # Make predictions
                y_pred = best_model.predict(X_test_processed)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train_processed, y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': best_model,
                    'MSE': mse,
                    'R2': r2,
                    'MAE': mae,
                    'CV_R2_mean': cv_mean,
                    'CV_R2_std': cv_std
                }
                
                print(f"   Test R²: {r2:.4f}")
                print(f"   Test MSE: {mse:.4f}")
                print(f"   Test MAE: {mae:.4f}")
                print(f"   CV R² (mean±std): {cv_mean:.4f}±{cv_std:.4f}")
                
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        return results, X_test_processed, y_test
    
    def create_ensemble_models(self, base_results, X_train, y_train):
        """Create ensemble models from best performing base models"""
        print("\n🎭 Creating Ensemble Models...")
        print("=" * 40)
        
        # Select top performing models for ensemble
        sorted_models = sorted(base_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        top_models = [(name, results['model']) for name, results in sorted_models[:5]]
        
        ensemble_results = {}
        
        # Voting Regressor
        voting_reg = VotingRegressor(estimators=top_models)
        voting_reg.fit(X_train, y_train)
        ensemble_results['VotingRegressor'] = voting_reg
        
        # Stacking Regressor
        stacking_reg = StackingRegressor(
            estimators=top_models,
            final_estimator=Ridge(alpha=10.0),
            cv=5
        )
        stacking_reg.fit(X_train, y_train)
        ensemble_results['StackingRegressor'] = stacking_reg
        
        return ensemble_results
    
    def fit(self, df=None):
        """Main training pipeline"""
        if df is None:
            df = self.load_and_preprocess_data()
        
        # Feature engineering
        df_enhanced = self.create_features(df)
        
        # Separate features and target
        X = df_enhanced.drop('hsc_result', axis=1)
        y = df_enhanced['hsc_result']
        
        # Create preprocessor
        self.preprocessor, numeric_features, categorical_features = self.create_preprocessor(X)
        
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Train and evaluate base models
        base_results, X_test_processed, y_test = self.train_and_evaluate_models(X, y)
        
        # Create ensemble models
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_processed = self.preprocessor.transform(X_train)
        ensemble_models = self.create_ensemble_models(base_results, X_train_processed, y_train)
        
        # Evaluate ensemble models
        print("\n🎭 Evaluating Ensemble Models...")
        for name, model in ensemble_models.items():
            y_pred = model.predict(X_test_processed)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            base_results[name] = {
                'model': model,
                'MSE': mse,
                'R2': r2,
                'MAE': mae,
                'CV_R2_mean': r2,  # Approximate
                'CV_R2_std': 0.0
            }
            
            print(f"🔍 {name}")
            print(f"   R²: {r2:.4f}")
            print(f"   MSE: {mse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print("-" * 40)
        
        # Store all results
        self.models = base_results
        
        # Find best model
        best_model_name = max(base_results.keys(), key=lambda k: base_results[k]['R2'])
        self.best_model = base_results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   R²: {base_results[best_model_name]['R2']:.4f}")
        print(f"   MSE: {base_results[best_model_name]['MSE']:.4f}")
        print(f"   MAE: {base_results[best_model_name]['MAE']:.4f}")
        
        return self
    
    def get_model_summary(self):
        """Get summary of all model performances"""
        if not self.models:
            return "No models trained yet. Call fit() first."
        
        summary = pd.DataFrame([
            {
                'Model': name,
                'R²': results['R2'],
                'MSE': results['MSE'],
                'MAE': results['MAE'],
                'CV_R2_mean': results.get('CV_R2_mean', 'N/A'),
                'CV_R2_std': results.get('CV_R2_std', 'N/A')
            }
            for name, results in self.models.items()
        ]).sort_values('R²', ascending=False)
        
        return summary

def main():
    """Main function to run improved model pipeline"""
    print("🚀 Student Performance Predictor - Improved Models")
    print("=" * 60)
    
    predictor = ImprovedStudentPerformancePredictor()
    predictor.fit()
    
    print("\n📊 Model Summary:")
    print("=" * 60)
    summary = predictor.get_model_summary()
    print(summary.to_string(index=False, float_format='%.4f'))
    
    return predictor

if __name__ == "__main__":
    predictor = main()