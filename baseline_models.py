import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def preprocess_data():
    """Load and preprocess the data similar to the notebook"""
    df = pd.read_csv('study.csv')
    
    # Drop the date column
    df = df.drop('29/04/2023', axis=1)
    
    # Rename columns to simpler names
    df = df.rename(columns={
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
    })
    
    # Handle missing values and data conversion
    df['time_spent_with_friends'] = pd.to_numeric(df['time_spent_with_friends'], errors='coerce')
    
    # Clean text data
    df['parent_status'] = df['parent_status'].str.strip() if 'parent_status' in df.columns else df['parent_status']
    df['smoker'] = df['smoker'].str.strip().str.capitalize() if 'smoker' in df.columns else df['smoker']
    df['relationship_breakdown'] = df['relationship_breakdown'].str.strip().str.capitalize()
    df['m_job'] = df['m_job'].str.strip().str.capitalize()
    df['f_job'] = df['f_job'].str.strip().str.capitalize()
    
    # Encode gender
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    
    # Apply log transformation to tuition cost
    df['average_tuition_fee_cost'] = np.log1p(df['average_tuition_fee_cost'])
    
    # Handle categorical variables with label encoding for now
    categorical_cols = ['address', 'family_size', 'parent_status', 'm_job', 'f_job', 'relationship_breakdown', 'smoker']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    
    return df

def evaluate_models():
    """Evaluate baseline models"""
    df = preprocess_data()
    
    X = df.drop('hsc_result', axis=1)
    y = df['hsc_result']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Dataset shape after preprocessing:", X.shape)
    print("Features:", X.columns.tolist())
    print("\nBaseline Model Evaluation:")
    print("=" * 50)
    
    # Tree-based models (don't need scaling)
    tree_models = {
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in tree_models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'MSE': mse, 'R2': r2, 'MAE': mae}
            
            print(f"🔍 {name}")
            print(f"   MSE: {mse:.4f}")
            print(f"   R² : {r2:.4f}")
            print(f"   MAE: {mae:.4f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Linear models (need scaling)
    linear_models = {
        "LinearRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        "Ridge": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(random_state=42))
        ]),
        "Lasso": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(random_state=42))
        ]),
        "ElasticNet": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(random_state=42))
        ]),
        "SVR": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR())
        ]),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', KNeighborsRegressor())
        ])
    }
    
    for name, model in linear_models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'MSE': mse, 'R2': r2, 'MAE': mae}
            
            print(f"🔍 {name}")
            print(f"   MSE: {mse:.4f}")
            print(f"   R² : {r2:.4f}")
            print(f"   MAE: {mae:.4f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['MSE'])
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   MSE: {best_model[1]['MSE']:.4f}")
    print(f"   R²:  {best_model[1]['R2']:.4f}")
    print(f"   MAE: {best_model[1]['MAE']:.4f}")
    
    return results

if __name__ == "__main__":
    results = evaluate_models()