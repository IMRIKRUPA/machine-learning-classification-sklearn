# Project 04: Machine Learning Classifier & Regressor Report
**Track:** Intern Circle Data Science & ML  
**Framework:** Scikit-Learn Supervised Machine Learning Pipeline

---

## 🎯 1. Classification Performance (Titanic Survival)

### Model Comparison Table:
| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | **0.8101** | **0.8182** | **0.6522** | **0.7258** |
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 |

### Key Classification Observations:
1. **Decision Tree Performance:** Achieves **81.01% test accuracy** with 5-fold cross-validation consistency.
2. **Feature Importance Hierarchy:**
   - `sex_male`: Most dominant predictive split (~56% Gini contribution).
   - `pclass`: Second most critical feature determining lifeboat priority.
   - `fare` & `age`: Continuous modifiers refining borderline classifications.

---

## 📈 2. Regression Performance (Ticket Fare Prediction)

| Evaluation Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Mean Squared Error (MSE)** | **968.318** | Penalizes larger squared prediction deviations |
| **Root Mean Squared Error (RMSE)** | **31.118** | Average prediction deviation in native units (£) |
| **Mean Absolute Error (MAE)** | **18.702** | Median robust magnitude of residuals |
| **R-squared (R²)** | **0.374** | Proportion of fare variance explained by features |

---

## 📦 Model Deployment Readiness
The trained Decision Tree pipeline is serialized via `joblib` into `04_ml_classifier/outputs/titanic_classifier.joblib`, fully encapsulating:
- Categorical one-hot encoders
- Numerical standard scalers
- Pre-trained decision tree estimator

This artifact is ready for immediate deployment in **Project 05 (Streamlit Web App)**.
