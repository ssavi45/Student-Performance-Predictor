"""
Production-Ready Student Performance Predictor

This module provides a simplified, production-ready implementation of the
improved student performance prediction model.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    """
    Production-ready Student Performance Predictor
    
    This class provides a complete pipeline for predicting HSC results
    based on student background and SSC performance.
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
    def preprocess_data(self, df):
        """
        Preprocess the raw data for model training/prediction
        
        Args:
            df: Raw DataFrame with student data
            
        Returns:
            Processed DataFrame ready for model input
        """
        df_clean = df.copy()
        
        # Drop timestamp column if present
        if '29/04/2023' in df_clean.columns:
            df_clean = df_clean.drop('29/04/2023', axis=1)
        
        # Standardize column names
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
            'avarage_tution_fee_cost': 'tuition_cost'
        }
        
        # Apply column mapping if columns exist
        df_clean = df_clean.rename(columns={k: v for k, v in column_mapping.items() if k in df_clean.columns})
        
        # Handle data types
        if 'time_spent_with_friends' in df_clean.columns:
            df_clean['time_with_friends'] = pd.to_numeric(df_clean['time_spent_with_friends'], errors='coerce')
            df_clean = df_clean.drop('time_spent_with_friends', axis=1)
        elif 'time_with_friends' not in df_clean.columns and 'time_spent_with_friends' in df_clean.columns:
            df_clean['time_with_friends'] = pd.to_numeric(df_clean['time_spent_with_friends'], errors='coerce')
        
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].map({'M': 1, 'F': 0})
        
        # Clean categorical variables
        categorical_cols = ['address', 'family_size', 'parent_status', 'm_job', 'f_job', 
                           'relationship_breakdown', 'smoker']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        
        return df_clean
    
    def engineer_features(self, df):
        """
        Create engineered features based on domain knowledge
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_eng = df.copy()
        
        # Key interaction features
        if 'ssc_result' in df_eng.columns:
            df_eng['ssc_squared'] = df_eng['ssc_result'] ** 2
            if 'age' in df_eng.columns:
                df_eng['ssc_age_interaction'] = df_eng['ssc_result'] * df_eng['age']
        
        # Education features
        if 'm_education' in df_eng.columns and 'f_education' in df_eng.columns:
            df_eng['education_sum'] = df_eng['m_education'] + df_eng['f_education']
            df_eng['education_max'] = df_eng[['m_education', 'f_education']].max(axis=1)
        
        # Family stability
        if 'parent_status' in df_eng.columns:
            df_eng['parent_together'] = (df_eng['parent_status'] == 't').astype(int)
        
        if 'relationship_breakdown' in df_eng.columns:
            df_eng['no_relationship_breakdown'] = (df_eng['relationship_breakdown'] == 'no').astype(int)
        
        if 'parent_together' in df_eng.columns and 'no_relationship_breakdown' in df_eng.columns:
            df_eng['family_stability'] = df_eng['parent_together'] * df_eng['no_relationship_breakdown']
        
        # Performance categories
        if 'ssc_result' in df_eng.columns:
            df_eng['ssc_high_performer'] = (df_eng['ssc_result'] >= 4.5).astype(int)
        
        # One-hot encode key categorical variables
        categorical_features = ['address', 'family_size', 'm_job', 'f_job', 'smoker']
        for col in categorical_features:
            if col in df_eng.columns:
                dummies = pd.get_dummies(df_eng[col], prefix=col, drop_first=True)
                df_eng = pd.concat([df_eng, dummies], axis=1)
                df_eng = df_eng.drop(col, axis=1)
        
        # Clean up intermediate columns
        cols_to_drop = ['parent_status', 'relationship_breakdown', 
                       'parent_together', 'no_relationship_breakdown']
        df_eng = df_eng.drop([col for col in cols_to_drop if col in df_eng.columns], axis=1)
        
        return df_eng
    
    def fit(self, data_path='study.csv'):
        """
        Train the model on the provided data
        
        Args:
            data_path: Path to the CSV file containing training data
        """
        print("🚀 Training Student Performance Predictor...")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Preprocess and engineer features
        df_processed = self.preprocess_data(df)
        df_engineered = self.engineer_features(df_processed)
        
        # Separate features and target
        if 'hsc_result' not in df_engineered.columns:
            raise ValueError("Target variable 'hsc_result' not found in data")
        
        X = df_engineered.drop('hsc_result', axis=1)
        y = df_engineered['hsc_result']
        
        self.feature_names = X.columns.tolist()
        print(f"Using {len(self.feature_names)} features for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model pipeline
        self.model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"✅ Model Training Complete!")
        print(f"   Test R²: {r2:.4f}")
        print(f"   Test MSE: {mse:.4f}")
        print(f"   Test MAE: {mae:.4f}")
        
        self.is_trained = True
        
        return self
    
    def predict(self, student_data):
        """
        Make predictions for new student data
        
        Args:
            student_data: DataFrame with student features or dict with student info
            
        Returns:
            Predicted HSC result(s)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        # Convert dict to DataFrame if necessary
        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])
        
        # Preprocess and engineer features
        df_processed = self.preprocess_data(student_data)
        df_engineered = self.engineer_features(df_processed)
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df_engineered.columns:
                df_engineered[feature] = 0  # Fill missing features with 0
        
        # Select only the features used in training
        X = df_engineered[self.feature_names]
        
        # Make prediction
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature importance from the RandomForest model
        regressor = self.model.named_steps['regressor']
        importance = regressor.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath='student_performance_model.joblib'):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='student_performance_model.joblib'):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model file
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Model loaded from {filepath}")
        return self

def main():
    """
    Demonstration of the StudentPerformancePredictor
    """
    print("🎓 Student Performance Predictor - Production Version")
    print("=" * 60)
    
    # Initialize and train the predictor
    predictor = StudentPerformancePredictor()
    predictor.fit('study.csv')
    
    # Show feature importance
    print("\n📊 Top 10 Important Features:")
    importance = predictor.get_feature_importance()
    print(importance.head(10).to_string(index=False, float_format='%.4f'))
    
    # Save the model
    predictor.save_model()
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    sample_student = {
        'gender': 1,  # Male
        'age': 18,
        'ssc_result': 4.5,
        'm_education': 3,
        'f_education': 3,
        'tuition_cost': 50000,
        'time_with_friends': 3,
        'address': 'u',
        'family_size': 'gt3',
        'smoker': 'no'
    }
    
    prediction = predictor.predict(sample_student)
    print(f"Predicted HSC Result: {prediction[0]:.3f}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()