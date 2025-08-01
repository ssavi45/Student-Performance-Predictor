import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             VotingRegressor, StackingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedStudentPerformancePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_and_clean_data(self):
        """Load and perform initial data cleaning"""
        df = pd.read_csv('study.csv')
        
        # Drop the timestamp column
        df = df.drop('29/04/2023', axis=1)
        
        # Simplify column names
        df.columns = [
            'gender', 'age', 'address', 'family_size', 'parent_status',
            'm_education', 'f_education', 'm_job', 'f_job', 
            'relationship_breakdown', 'smoker', 'tuition_cost',
            'time_with_friends', 'ssc_result', 'hsc_result'
        ]
        
        # Data type conversions and cleaning
        df['time_with_friends'] = pd.to_numeric(df['time_with_friends'], errors='coerce')
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
        
        # Clean categorical variables
        categorical_cols = ['address', 'family_size', 'parent_status', 'm_job', 'f_job', 
                           'relationship_breakdown', 'smoker']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def advanced_feature_engineering(self, df):
        """Create comprehensive feature engineering"""
        df_new = df.copy()
        
        # 1. Statistical features from education levels
        df_new['education_sum'] = df_new['m_education'] + df_new['f_education']
        df_new['education_diff'] = abs(df_new['m_education'] - df_new['f_education'])
        df_new['education_max'] = df_new[['m_education', 'f_education']].max(axis=1)
        df_new['education_min'] = df_new[['m_education', 'f_education']].min(axis=1)
        
        # 2. Interaction terms with SSC result (most important predictor)
        df_new['ssc_age_ratio'] = df_new['ssc_result'] / df_new['age']
        df_new['ssc_education_product'] = df_new['ssc_result'] * df_new['education_sum']
        df_new['ssc_tuition_ratio'] = df_new['ssc_result'] / (df_new['tuition_cost'] + 1)
        
        # 3. Polynomial features for key variables
        df_new['ssc_squared'] = df_new['ssc_result'] ** 2
        df_new['ssc_cubed'] = df_new['ssc_result'] ** 3
        df_new['age_squared'] = df_new['age'] ** 2
        
        # 4. Social and family factors
        df_new['social_stability'] = (
            (df_new['parent_status'] == 't') & 
            (df_new['relationship_breakdown'] == 'no')
        ).astype(int)
        
        df_new['healthy_lifestyle'] = (df_new['smoker'] == 'no').astype(int)
        
        # 5. Study environment factors
        df_new['study_time_per_cost'] = df_new['time_with_friends'] / (df_new['tuition_cost'] + 1)
        df_new['family_education_support'] = df_new['education_sum'] * df_new['social_stability']
        
        # 6. Performance categories
        df_new['ssc_performance_level'] = pd.cut(
            df_new['ssc_result'], 
            bins=[0, 3.5, 4.2, 4.8, 5.0], 
            labels=[0, 1, 2, 3]
        ).astype(float)
        
        # 7. Age group effects
        df_new['age_group'] = pd.cut(df_new['age'], bins=[0, 18, 20, 25], labels=[0, 1, 2]).astype(float)
        
        # 8. Economic factors
        df_new['tuition_burden'] = df_new['tuition_cost'] / (df_new['education_sum'] + 1)
        
        # 9. One-hot encode remaining categorical variables
        categorical_features = ['address', 'family_size', 'parent_status', 'm_job', 'f_job', 
                               'relationship_breakdown', 'smoker']
        
        for col in categorical_features:
            if col in df_new.columns:
                dummies = pd.get_dummies(df_new[col], prefix=col, drop_first=True)
                df_new = pd.concat([df_new, dummies], axis=1)
                df_new.drop(col, axis=1, inplace=True)
        
        return df_new
    
    def create_advanced_pipeline(self, X, y):
        """Create advanced preprocessing pipeline with feature selection"""
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop'))
        
        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Feature scaling and power transformation
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train_transformed = pd.DataFrame(
            power_transformer.fit_transform(X_train_imputed),
            columns=X_train_imputed.columns,
            index=X_train_imputed.index
        )
        X_test_transformed = pd.DataFrame(
            power_transformer.transform(X_test_imputed),
            columns=X_test_imputed.columns,
            index=X_test_imputed.index
        )
        
        # Feature selection using multiple methods
        # 1. Statistical selection
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X_train_transformed, y_train)
        feature_scores = pd.DataFrame({
            'feature': X_train_transformed.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        print("Top 10 features by statistical score:")
        print(feature_scores.head(10))
        
        # Select top features
        top_k = min(20, len(X_train_transformed.columns))  # Select top 20 or all if less
        top_features = feature_scores.head(top_k)['feature'].tolist()
        
        X_train_selected = X_train_transformed[top_features]
        X_test_selected = X_test_transformed[top_features]
        
        return X_train_selected, X_test_selected, y_train, y_test, top_features
    
    def create_optimized_models(self):
        """Create models with optimized hyperparameters"""
        models = {
            # Ensemble methods with optimized parameters
            'OptimizedRandomForest': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'OptimizedXGBoost': XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='rmse'
            ),
            
            'OptimizedLightGBM': LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            ),
            
            'OptimizedCatBoost': CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0
            ),
            
            # Advanced linear models
            'BayesianRidge': BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            ),
            
            'OptimizedElasticNet': ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                random_state=42
            ),
            
            # Neural network with optimized architecture
            'OptimizedMLP': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            
            # Support Vector Machine
            'OptimizedSVR': SVR(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                epsilon=0.01
            )
        }
        
        return models
    
    def create_meta_ensemble(self, base_models, X_train, y_train):
        """Create advanced ensemble methods"""
        
        # Create base model list for ensemble
        estimators = [(name, model) for name, model in base_models.items() 
                     if hasattr(model, 'predict')]
        
        ensemble_models = {}
        
        # Voting ensemble with weights based on performance
        voting_regressor = VotingRegressor(
            estimators=estimators,
            weights=None  # Equal weights initially
        )
        ensemble_models['VotingEnsemble'] = voting_regressor
        
        # Stacking ensemble with multiple meta-learners
        for meta_name, meta_model in [
            ('Ridge', Ridge(alpha=1.0)),
            ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5)),
            ('XGBoost', XGBRegressor(n_estimators=100, max_depth=3, random_state=42))
        ]:
            stacking_regressor = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=5,
                passthrough=False
            )
            ensemble_models[f'Stacking_{meta_name}'] = stacking_regressor
        
        return ensemble_models
    
    def evaluate_model_with_cv(self, model, X, y, model_name, cv_folds=5):
        """Comprehensive model evaluation with cross-validation"""
        
        # K-fold cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        cv_mse_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        cv_mae_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
        
        results = {
            'model': model,
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_r2_std': cv_r2_scores.std(),
            'cv_mse_mean': cv_mse_scores.mean(),
            'cv_mse_std': cv_mse_scores.std(),
            'cv_mae_mean': cv_mae_scores.mean(),
            'cv_mae_std': cv_mae_scores.std()
        }
        
        print(f"🔍 {model_name}")
        print(f"   CV R² : {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        print(f"   CV MSE: {results['cv_mse_mean']:.4f} ± {results['cv_mse_std']:.4f}")
        print(f"   CV MAE: {results['cv_mae_mean']:.4f} ± {results['cv_mae_std']:.4f}")
        print("-" * 50)
        
        return results
    
    def fit(self):
        """Main training pipeline with advanced techniques"""
        print("🚀 Advanced Student Performance Predictor")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_and_clean_data()
        df_enhanced = self.advanced_feature_engineering(df)
        
        # Separate features and target
        X = df_enhanced.drop('hsc_result', axis=1)
        y = df_enhanced['hsc_result']
        
        print(f"Features after engineering: {X.shape[1]}")
        print(f"Feature names: {X.columns.tolist()}")
        
        # Advanced preprocessing with feature selection
        X_train, X_test, y_train, y_test, selected_features = self.create_advanced_pipeline(X, y)
        
        print(f"\nSelected {len(selected_features)} features for modeling")
        
        # Create and evaluate base models
        print("\n📊 Base Model Evaluation (Cross-Validation)")
        print("=" * 60)
        
        base_models = self.create_optimized_models()
        base_results = {}
        
        for name, model in base_models.items():
            try:
                # Fit the model
                model.fit(X_train, y_train)
                
                # Cross-validation evaluation
                results = self.evaluate_model_with_cv(model, X_train, y_train, name)
                base_results[name] = results
                
                # Test set evaluation
                y_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, y_pred)
                test_mse = mean_squared_error(y_test, y_pred)
                test_mae = mean_absolute_error(y_test, y_pred)
                
                base_results[name]['test_r2'] = test_r2
                base_results[name]['test_mse'] = test_mse
                base_results[name]['test_mae'] = test_mae
                
                print(f"   Test R²:  {test_r2:.4f}")
                print(f"   Test MSE: {test_mse:.4f}")
                print(f"   Test MAE: {test_mae:.4f}")
                print("=" * 50)
                
            except Exception as e:
                print(f"   Error training {name}: {e}")
                continue
        
        # Create and evaluate ensemble models
        print("\n🎭 Ensemble Model Evaluation")
        print("=" * 60)
        
        # Select top 5 models for ensemble
        top_models = sorted(base_results.items(), 
                           key=lambda x: x[1]['cv_r2_mean'], 
                           reverse=True)[:5]
        
        top_base_models = {name: results['model'] for name, results in top_models}
        
        ensemble_models = self.create_meta_ensemble(top_base_models, X_train, y_train)
        ensemble_results = {}
        
        for name, model in ensemble_models.items():
            try:
                # Fit ensemble model
                model.fit(X_train, y_train)
                
                # Evaluate ensemble
                results = self.evaluate_model_with_cv(model, X_train, y_train, name)
                ensemble_results[name] = results
                
                # Test set evaluation
                y_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, y_pred)
                test_mse = mean_squared_error(y_test, y_pred)
                test_mae = mean_absolute_error(y_test, y_pred)
                
                ensemble_results[name]['test_r2'] = test_r2
                ensemble_results[name]['test_mse'] = test_mse
                ensemble_results[name]['test_mae'] = test_mae
                
                print(f"   Test R²:  {test_r2:.4f}")
                print(f"   Test MSE: {test_mse:.4f}")
                print(f"   Test MAE: {test_mae:.4f}")
                print("=" * 50)
                
            except Exception as e:
                print(f"   Error training {name}: {e}")
                continue
        
        # Combine all results
        all_results = {**base_results, **ensemble_results}
        self.models = all_results
        
        # Find best model based on test R²
        if all_results:
            best_model_name = max(all_results.keys(), 
                                key=lambda k: all_results[k].get('test_r2', -np.inf))
            self.best_model = all_results[best_model_name]['model']
            
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   Test R²:  {all_results[best_model_name]['test_r2']:.4f}")
            print(f"   Test MSE: {all_results[best_model_name]['test_mse']:.4f}")
            print(f"   Test MAE: {all_results[best_model_name]['test_mae']:.4f}")
            print(f"   CV R²:    {all_results[best_model_name]['cv_r2_mean']:.4f} ± {all_results[best_model_name]['cv_r2_std']:.4f}")
        
        return self
    
    def get_model_summary(self):
        """Get comprehensive model performance summary"""
        if not self.models:
            return "No models trained yet. Call fit() first."
        
        summary_data = []
        for name, results in self.models.items():
            summary_data.append({
                'Model': name,
                'Test_R²': results.get('test_r2', 'N/A'),
                'Test_MSE': results.get('test_mse', 'N/A'),
                'Test_MAE': results.get('test_mae', 'N/A'),
                'CV_R²_Mean': results.get('cv_r2_mean', 'N/A'),
                'CV_R²_Std': results.get('cv_r2_std', 'N/A'),
                'CV_MSE_Mean': results.get('cv_mse_mean', 'N/A')
            })
        
        summary_df = pd.DataFrame(summary_data)
        if 'Test_R²' in summary_df.columns:
            summary_df = summary_df.sort_values('Test_R²', ascending=False)
        
        return summary_df

def main():
    """Main execution function"""
    predictor = AdvancedStudentPerformancePredictor()
    predictor.fit()
    
    print("\n📈 Final Model Performance Summary")
    print("=" * 80)
    summary = predictor.get_model_summary()
    print(summary.to_string(index=False, float_format='%.4f'))
    
    return predictor

if __name__ == "__main__":
    predictor = main()