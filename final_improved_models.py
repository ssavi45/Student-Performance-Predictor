import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FinalStudentPerformancePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess data with the best practices identified"""
        df = pd.read_csv('study.csv')
        
        # Drop timestamp column
        df = df.drop('29/04/2023', axis=1)
        
        # Simplify column names
        df.columns = ['gender', 'age', 'address', 'family_size', 'parent_status',
                      'm_education', 'f_education', 'm_job', 'f_job', 
                      'relationship_breakdown', 'smoker', 'tuition_cost',
                      'time_with_friends', 'ssc_result', 'hsc_result']
        
        # Handle data types
        df['time_with_friends'] = pd.to_numeric(df['time_with_friends'], errors='coerce')
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
        
        # Clean categorical variables
        categorical_cols = ['address', 'family_size', 'parent_status', 'm_job', 'f_job', 
                           'relationship_breakdown', 'smoker']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        return df
    
    def engineer_features(self, df):
        """Apply the most effective feature engineering techniques"""
        df_new = df.copy()
        
        # Key interaction terms (from analysis, SSC result is most important)
        df_new['ssc_squared'] = df_new['ssc_result'] ** 2
        df_new['ssc_age_interaction'] = df_new['ssc_result'] * df_new['age']
        
        # Education features
        df_new['education_sum'] = df_new['m_education'] + df_new['f_education']
        df_new['education_max'] = df_new[['m_education', 'f_education']].max(axis=1)
        
        # Family stability
        df_new['parent_together'] = (df_new['parent_status'] == 't').astype(int)
        df_new['no_relationship_breakdown'] = (df_new['relationship_breakdown'] == 'no').astype(int)
        df_new['family_stability'] = df_new['parent_together'] * df_new['no_relationship_breakdown']
        
        # Performance categories
        df_new['ssc_high_performer'] = (df_new['ssc_result'] >= 4.5).astype(int)
        
        # One-hot encode key categorical variables
        important_categoricals = ['address', 'family_size', 'm_job', 'f_job', 'smoker']
        for col in important_categoricals:
            if col in df_new.columns:
                dummies = pd.get_dummies(df_new[col], prefix=col, drop_first=True)
                df_new = pd.concat([df_new, dummies], axis=1)
                df_new.drop(col, axis=1, inplace=True)
        
        # Drop redundant columns
        cols_to_drop = ['parent_status', 'relationship_breakdown', 
                       'parent_together', 'no_relationship_breakdown']
        df_new = df_new.drop([col for col in cols_to_drop if col in df_new.columns], axis=1)
        
        return df_new
    
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
    def create_models(self):
        """Create optimized models based on our analysis"""
        models = {
            # Best performing tree-based model with tuned parameters
            'OptimizedRandomForest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            
            # Gradient Boosting with moderate complexity
            'OptimizedGradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            
            # XGBoost with regularization
            'OptimizedXGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='rmse'
            ),
            
            # LightGBM optimized
            'OptimizedLightGBM': LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                num_leaves=40,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            ),
            
            # Ridge regression as a stable baseline
            'OptimizedRidge': Ridge(alpha=1.0),
            
            # Elastic Net for feature selection
            'OptimizedElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        }
        
        return models
    
    def train_and_evaluate(self, X, y):
        """Train models and evaluate with cross-validation"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit preprocessor
        self.preprocessor = self.create_preprocessing_pipeline()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        models = self.create_models()
        results = {}
        
        print("🔍 Model Training and Evaluation")
        print("=" * 60)
        
        for name, model in models.items():
            try:
                # Fit model
                model.fit(X_train_processed, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_processed, y_train, 
                                          cv=5, scoring='r2')
                
                # Test predictions
                y_pred = model.predict(X_test_processed)
                
                # Metrics
                test_r2 = r2_score(y_test, y_pred)
                test_mse = mean_squared_error(y_test, y_pred)
                test_mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'predictions': y_pred
                }
                
                print(f"{name}:")
                print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"  Test R²: {test_r2:.4f}")
                print(f"  Test MSE: {test_mse:.4f}")
                print(f"  Test MAE: {test_mae:.4f}")
                print("-" * 40)
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        # Create ensemble model
        if len(results) >= 3:
            print("\n🎭 Creating Ensemble Model")
            print("=" * 40)
            
            # Select top 3 models for ensemble
            top_models = sorted(results.items(), 
                              key=lambda x: x[1]['test_r2'], 
                              reverse=True)[:3]
            
            ensemble_estimators = [(name, res['model']) for name, res in top_models]
            
            stacking_model = StackingRegressor(
                estimators=ensemble_estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5
            )
            
            stacking_model.fit(X_train_processed, y_train)
            
            # Evaluate ensemble
            cv_scores = cross_val_score(stacking_model, X_train_processed, y_train, 
                                      cv=5, scoring='r2')
            y_pred_ensemble = stacking_model.predict(X_test_processed)
            
            ensemble_r2 = r2_score(y_test, y_pred_ensemble)
            ensemble_mse = mean_squared_error(y_test, y_pred_ensemble)
            ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
            
            results['StackingEnsemble'] = {
                'model': stacking_model,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'test_r2': ensemble_r2,
                'test_mse': ensemble_mse,
                'test_mae': ensemble_mae,
                'predictions': y_pred_ensemble
            }
            
            print(f"Stacking Ensemble:")
            print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Test R²: {ensemble_r2:.4f}")
            print(f"  Test MSE: {ensemble_mse:.4f}")
            print(f"  Test MAE: {ensemble_mae:.4f}")
        
        self.models = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"  Test R²: {results[best_model_name]['test_r2']:.4f}")
        print(f"  Test MSE: {results[best_model_name]['test_mse']:.4f}")
        print(f"  Test MAE: {results[best_model_name]['test_mae']:.4f}")
        
        return results, X_test, y_test
    
    def analyze_feature_importance(self, X):
        """Analyze feature importance from the best model"""
        if self.best_model is None:
            return None
        
        # For tree-based models, get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Feature Importance (Top 10):")
            print("=" * 40)
            print(feature_importance_df.head(10).to_string(index=False, float_format='%.4f'))
            
            return feature_importance_df
        
        return None
    
    def create_performance_visualization(self, X_test, y_test):
        """Create visualization of model performance"""
        if not self.models:
            return
        
        # Get predictions from best model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['test_r2'])
        y_pred = self.models[best_model_name]['predictions']
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual HSC Result')
        axes[0, 0].set_ylabel('Predicted HSC Result')
        axes[0, 0].set_title(f'Actual vs Predicted ({best_model_name})')
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted HSC Result')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # 3. Model comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['test_r2'] for name in model_names]
        
        axes[1, 0].barh(model_names, r2_scores)
        axes[1, 0].set_xlabel('R² Score')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].set_xlim(min(r2_scores) - 0.01, max(r2_scores) + 0.01)
        
        # 4. Distribution of predictions vs actual
        axes[1, 1].hist(y_test, bins=20, alpha=0.5, label='Actual', density=True)
        axes[1, 1].hist(y_pred, bins=20, alpha=0.5, label='Predicted', density=True)
        axes[1, 1].set_xlabel('HSC Result')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📈 Performance visualization saved as 'model_performance_analysis.png'")
    
    def fit(self):
        """Main training pipeline"""
        print("🚀 Final Student Performance Predictor")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        df_engineered = self.engineer_features(df)
        
        # Separate features and target
        X = df_engineered.drop('hsc_result', axis=1)
        y = df_engineered['hsc_result']
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features after engineering: {X.shape[1]}")
        print(f"Target distribution: mean={y.mean():.3f}, std={y.std():.3f}")
        
        # Train and evaluate models
        results, X_test, y_test = self.train_and_evaluate(X, y)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance(X)
        
        # Create visualization
        self.create_performance_visualization(X_test, y_test)
        
        return self
    
    def get_summary(self):
        """Get model performance summary"""
        if not self.models:
            return "No models trained yet."
        
        summary_data = []
        for name, results in self.models.items():
            summary_data.append({
                'Model': name,
                'CV_R²_Mean': results['cv_r2_mean'],
                'CV_R²_Std': results['cv_r2_std'],
                'Test_R²': results['test_r2'],
                'Test_MSE': results['test_mse'],
                'Test_MAE': results['test_mae']
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('Test_R²', ascending=False)
        return summary_df

def main():
    """Main execution function"""
    predictor = FinalStudentPerformancePredictor()
    predictor.fit()
    
    print("\n📊 Final Performance Summary")
    print("=" * 80)
    summary = predictor.get_summary()
    print(summary.to_string(index=False, float_format='%.4f'))
    
    return predictor

if __name__ == "__main__":
    predictor = main()