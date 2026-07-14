# Student Academic Performance Predictor

This repository contains a machine learning pipeline designed to predict secondary and higher secondary academic outcomes (specifically HSC grades) using prior academic performance (such as SSC GPA) and socio-academic background features. By exploring both regression (continuous GPA prediction) and class-balanced classification (discrete grade prediction) architectures, the project identifies key factors that influence academic success and provides a robust framework for forecasting student outcomes.

---

### The Machine Learning Pipeline

The project codebase explores two main paradigms: continuous GPA regression in [study.ipynb](study.ipynb) / [Regression/main.ipynb](Regression/main.ipynb) and discrete grade classification in [final.ipynb](final.ipynb).

* **Data Preprocessing**:
    * **Feature Selection**: Dropped administrative and non-predictive features (such as the date column) and sparse categorical parental job fields (`m_job`, `f_job`).
    * **Imputation**: Replaced missing values in `time_spent_with_friends` using the rounded feature mean.
    * **Categorical Encoding**: Mapped binary categorical variables (e.g., `gender`, `address`, `fam_size`, `parent_status`, `relationship_breakdown`, `smoker`) into binary numerical format to avoid scale bias.
* **Feature Engineering**:
    * **Skewness Correction**: Applied a log transformation (`np.log1p`) to the highly skewed tuition fee cost feature (`avarage_tution_fee_cost`).
    * **Target Discretization**: Converted the continuous target `hsc_result` (GPA out of 5.0) into discrete grade categories (`hsc_grade`) using the custom function `gpa_to_grade` inside [final.ipynb](final.ipynb) to handle non-linear boundaries.
    * **Class Imbalance Mitigation**: Integrated Random Over-Sampling (`RandomOverSampler` from `imblearn`) into the training pipelines to address the heavily skewed grade distribution (where the `A` grade represents ~90% of samples).
    * **Cross-Validation**: Implemented 5-fold Stratified K-Fold cross-validation to ensure reliable evaluation across minority classes.
* **Model Exploration**:
    * **Regression**: Evaluated Linear Regression, Ridge, Lasso, ElasticNet, K-Nearest Neighbors (KNN), Support Vector Regression (SVR), Principal Component Regression (PCR), and tree-based ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost).
    * **Classification**: Evaluated Logistic Regression, Support Vector Classifier (SVC), Naive Bayes, Decision Trees, Random Forest, Extra Trees, and ensemble architectures (Bagging, AdaBoost, Gradient Boosting, XGBoost, Stacking Classifiers).

---

### Core Performance & Metrics

#### 1. Classification (HSC Grade Prediction)
Due to severe class imbalance, the models in [final.ipynb](final.ipynb) were evaluated using macro-averaged metrics. The final champion model is **AdaBoost (with a Decision Tree base estimator)** trained with Random Over-Sampling (ROS).

| Model | Configuration | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Cohen's Kappa | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost (DT)** | Ensembles (ROS) | 0.8659 | 0.6334 | 0.6345 | **0.6339** | 0.3173 | 0.3175 |
| **XGBoost Classifier** | Ensembles (ROS) | 0.8800 | 0.6028 | 0.6376 | 0.6189 | 0.3368 | 0.3377 |
| **Random Forest** | Baseline (No ROS) | **0.9247** | 0.5874 | 0.6432 | 0.6139 | 0.4618 | 0.5196 |
| **Bagging (DT)** | Ensembles (ROS) | 0.9129 | 0.5628 | 0.5989 | 0.5803 | 0.3860 | 0.4280 |
| **Extra Trees** | Baseline | 0.9106 | 0.5778 | 0.5182 | 0.5396 | 0.2999 | 0.3741 |
| **SVM** | Robust Scaler | 0.9129 | 0.5779 | 0.5191 | 0.5400 | 0.3077 | 0.3948 |
| **Logistic Regression** | Baseline | 0.9082 | 0.4078 | 0.4030 | 0.4043 | 0.3148 | 0.3697 |
| **Gaussian Naive Bayes** | Baseline | 0.8847 | 0.3810 | 0.4263 | 0.4015 | 0.3102 | 0.3201 |

#### 2. Regression (HSC GPA Prediction)
For the continuous prediction of final GPA, baseline results are documented in [study.ipynb](study.ipynb). The low $R^2$ scores across all regressors demonstrated the highly non-linear relationship of features to exact numeric outcomes, prompting the transition to the categorical grade classification architecture shown above.

| Regressor | Mean Squared Error (MSE) | $R^2$ Score |
| :--- | :---: | :---: |
| **Gradient Boosting** | 0.0841 | **0.0873** |
| **AdaBoost** | 0.0848 | 0.0798 |
| **CatBoost** | 0.0886 | 0.0383 |
| **Random Forest** | 0.0896 | 0.0271 |
| **Linear Regression / Ridge** | 0.0900 | 0.0226 |
| **LightGBM** | 0.0932 | -0.0117 |
| **SVR** | 0.0937 | -0.0174 |
| **XGBoost** | 0.1101 | -0.1954 |

---

### Tech Stack

* **Language**: Python 3.12+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, Imbalanced-Learn (`imblearn`)
* **Gradient Boosting**: XGBoost, LightGBM, CatBoost
* **Visualization & Interpretability**: Matplotlib, Seaborn, SHAP

---

### Quick Replication Guide

Follow these steps to set up the environment and reproduce the evaluations:

```bash
# 1. Navigate to the project root directory
cd Student-Performance-Predictor-main

# 2. Set up a virtual environment (Python 3.12+ recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install core dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost matplotlib seaborn ipykernel shap

# 4. Open the Jupyter Notebook environment
jupyter notebook

# 5. Run the final evaluation pipeline
# Open 'final.ipynb' and execute all cells to train and evaluate the classification models.
