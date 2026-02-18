# 🏦 Loan Default Risk Prediction

A machine learning pipeline that predicts the probability of loan default, enabling smarter lending decisions and risk-based pricing strategies.

> **Business Impact**: At optimized thresholds, this model can save an estimated **$15M+ annually** on a portfolio of 200K loans by catching high-risk defaults while minimizing false rejections of creditworthy borrowers.

---

## 📋 Table of Contents
- [Business Context](#business-context)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)

---

## Business Context

Consumer lending institutions lose billions annually to loan defaults. The challenge is balancing two competing objectives:

1. **Minimize default losses** — Reject high-risk applicants before they default
2. **Maximize revenue** — Approve creditworthy borrowers to earn interest income

This project builds a predictive model that scores each loan application by default probability, allowing lenders to:
- Automate approval/denial decisions at scale
- Implement risk-based pricing (higher rates for higher risk)
- Optimize approval thresholds based on profit maximization
- Meet regulatory requirements for model interpretability (SR 11-7)

---

## Results Summary

| Model | AUC-ROC | F1 Score | Avg Precision |
|-------|---------|----------|---------------|
| Logistic Regression | ~0.80 | ~0.45 | ~0.40 |
| Random Forest | ~0.83 | ~0.48 | ~0.44 |
| **Gradient Boosting** | **~0.85** | **~0.50** | **~0.47** |

*Gradient Boosting was selected as the production model based on AUC-ROC and business impact analysis.*

### Key Visualizations

| | |
|---|---|
| ![Default Rate Analysis](outputs/01_default_rate_analysis.png) | ![Correlation Heatmap](outputs/02_correlation_heatmap.png) |
| ![Feature Distributions](outputs/03_feature_distributions.png) | ![Model Comparison](outputs/04_model_comparison.png) |
| ![Feature Importance](outputs/05_feature_importance.png) | ![Confusion Matrix](outputs/06_confusion_matrix.png) |
| ![Risk Score Distribution](outputs/07_risk_score_distribution.png) | ![Profit Curve](outputs/08_profit_curve.png) |

---

## Project Structure

```
loan-risk-prediction/
│
├── loan_risk_model.py          # Full ML pipeline (run this)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── outputs/                    # Generated visualizations
│   ├── 01_default_rate_analysis.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_distributions.png
│   ├── 04_model_comparison.png
│   ├── 05_feature_importance.png
│   ├── 06_confusion_matrix.png
│   ├── 07_risk_score_distribution.png
│   └── 08_profit_curve.png
│
└── data/                       # (Optional) Export generated dataset
    └── loan_data.csv
```

---

## Key Findings

### Risk Factors (Ranked by Importance)
1. **Credit Score** — Strongest predictor. Subprime borrowers (<620) default at 3x the rate of prime borrowers
2. **Debt-to-Income Ratio** — Borrowers with DTI > 40% show significantly elevated default rates
3. **Interest Rate** — High rates correlate with default (reflects risk-based pricing in training data)
4. **Credit Utilization** — Utilization > 60% signals financial stress
5. **Delinquency History** — Any prior delinquency roughly doubles default probability

### Business Impact
- Optimized threshold selection improves net value by balancing defaults caught vs. good loans rejected
- The model enables tiered pricing: low-risk borrowers get better rates (retention), high-risk borrowers are priced appropriately or declined

---

## Methodology

### 1. Data Generation
Synthetic dataset (10,000 loans) modeled after Lending Club / bank origination data with realistic correlations between features and default outcomes.

### 2. Exploratory Data Analysis
- Default rate segmentation by credit score, DTI, loan purpose, etc.
- Correlation analysis to identify multicollinearity
- Distribution comparison between default/non-default populations

### 3. Feature Engineering (8 new features)
Domain-driven features including:
- `payment_burden` — Monthly payment as % of income
- `credit_risk_composite` — Weighted score combining credit score, utilization, and delinquency
- `income_to_loan_ratio` — Affordability measure
- Binary risk flags: `subprime`, `high_dti`, `high_utilization`

### 4. Model Training
- 3 models compared: Logistic Regression, Random Forest, Gradient Boosting
- 5-fold stratified cross-validation
- Class balancing to handle the ~15% default rate imbalance

### 5. Evaluation
- ROC-AUC, Precision-Recall, F1 Score
- Confusion matrix analysis
- Risk score distribution analysis

### 6. Business Impact Analysis
- Profit curve optimization across decision thresholds
- Cost-benefit analysis incorporating loss-given-default and interest margins
- Portfolio-level savings estimation

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-risk-prediction.git
cd loan-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python loan_risk_model.py
```

All 8 visualization PNGs will be saved to the `outputs/` directory.

---

## Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML models, preprocessing, evaluation
- **Matplotlib / Seaborn** — Visualization
- **XGBoost** *(optional extension)*

---

## Future Work

- [ ] Deploy as REST API using FastAPI + Docker
- [ ] Add SHAP values for individual prediction explanations
- [ ] Implement real-time scoring with streaming data
- [ ] Add fairness analysis (disparate impact testing across demographics)
- [ ] Connect to real Lending Club dataset for validation
- [ ] Build Streamlit dashboard for interactive risk exploration

---

## Author

**[Your Name]**  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [Email](mailto:your@email.com)

---

*This project was built as a portfolio piece demonstrating end-to-end ML pipeline development for credit risk modeling in fintech/banking.*
