# Auto Insurance Fraud Detection - Comparing different models:

## Overview
This project focuses on detecting fraudulent auto insurance claims using **different machine learning algorithms**. The goal is to compare their performance and identify the most effective model for fraud detection in imbalanced datasets.

## Objective
- Build and evaluate multiple supervised learning models.  
- Compare them using different performance metrics.  
- Highlight the trade-off between accuracy and the ability to correctly identify fraudulent claims.  

## Dataset
- Source: Auto Insurance Claims Dataset
- Features include customer, vehicle, and claim-related attributes.  
- Target variable: **fraud_reported** (Yes/No).

### Data Preprocessing
- Removed irrelevant features (e.g., `policy_number`, `incident_date`).  
- Encoded categorical variables.  
- Handled missing values.  
- Train/test split (and cross-validation for model selection).  
- Applied class balancing techniques (when necessary).  

## Models Trained
The following algorithms were compared:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  
- CatBoost  
- Extra Trees  
- LightGBM  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Voting Classifier (Ensemble of best models)  

## Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC  
- **Hyperparameter tuning:** Performed with `GridSearchCV` and cross-validation.  
- **Key findings:**  
  - Ensemble methods (XGBoost, LightGBM, Random Forest) performed better than simpler models.  
  - Logistic Regression provided interpretability but lower recall.  

## Results
- **Best model:** `Best Model Name` with `Metrics`  
- Fraud detection is a highly imbalanced problem; thus, Recall and F1-score were prioritized over Accuracy.  
- Feature importance analysis revealed that attributes like `Top Features` were most significant in predicting fraud.

## Conclusion
- Ensemble-based models (XGBoost, LightGBM, Random Forest) achieved the best balance between sensitivity and specificity.  
- Interpretability tools (e.g., SHAP, LIME) could further explain model decisions.  
- The project shows the importance of handling class imbalance and evaluating beyond accuracy.  

## Future Improvements
- Apply advanced resampling techniques (SMOTE, ADASYN, undersampling).  
- Explore cost-sensitive learning (e.g., focal loss).  
- Use SHAP/LIME for model explainability.  
- Consider temporal validation if the dataset contains time-related features.

## Requirements
- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- lightgbm  
- catboost  
- matplotlib  
- seaborn  
- jupyter  
- ipykernel  


## How to Run
```bash
# Clone repository
git clone https://github.com/igorambrozini/Insurance_fraud_detection_many_classification_models.git
cd Insurance_fraud_detection_many_classification_models

# Activate virtual environment
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/insurance_fraud_detection.ipynb
