"""
Loan Default Risk Prediction Model
====================================
A machine learning pipeline for predicting loan default risk.
Built as a portfolio project for fintech/banking data science roles.

Author: [Your Name]
Date: 2026
Tech Stack: Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn

Business Context:
    Banks lose billions annually to loan defaults. This model predicts
    the probability of a borrower defaulting, enabling better lending
    decisions and risk-based pricing strategies.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA GENERATION (Simulated Realistic Loan Data)
# ============================================================
# In a real project, you'd load from a database or CSV.
# This synthetic data mirrors Lending Club / bank loan datasets.

def generate_loan_data(n_samples=10000, random_state=42):
    """
    Generate realistic synthetic loan application data.
    
    Features mirror real-world lending data:
    - Borrower demographics (age, income, employment)
    - Loan characteristics (amount, term, interest rate)
    - Credit history (credit score, delinquencies, utilization)
    """
    np.random.seed(random_state)
    
    # Borrower Demographics
    age = np.random.normal(38, 12, n_samples).clip(21, 75).astype(int)
    annual_income = np.random.lognormal(10.8, 0.7, n_samples).clip(15000, 500000).astype(int)
    employment_length = np.random.exponential(5, n_samples).clip(0, 40).astype(int)
    
    # Credit History
    credit_score = np.random.normal(680, 80, n_samples).clip(300, 850).astype(int)
    num_credit_lines = np.random.poisson(8, n_samples).clip(1, 30)
    credit_utilization = np.random.beta(2, 5, n_samples).clip(0, 1)
    num_delinquencies = np.random.poisson(0.3, n_samples).clip(0, 10)
    months_since_delinquency = np.where(
        num_delinquencies > 0,
        np.random.exponential(24, n_samples).clip(1, 120).astype(int),
        np.nan
    )
    
    # Loan Characteristics
    loan_amount = np.random.lognormal(9.5, 0.8, n_samples).clip(1000, 100000).astype(int)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.05, 0.15, 0.40, 0.25, 0.15])
    
    # Interest rate (correlated with credit score)
    base_rate = 15 - (credit_score - 300) / 550 * 12
    interest_rate = (base_rate + np.random.normal(0, 1.5, n_samples)).clip(3.5, 28.0)
    
    # Derived Features
    debt_to_income = (loan_amount / loan_term_months * 12) / annual_income
    debt_to_income = debt_to_income.clip(0, 1)
    
    loan_purpose = np.random.choice(
        ['debt_consolidation', 'home_improvement', 'major_purchase', 
         'medical', 'small_business', 'education', 'auto'],
        n_samples,
        p=[0.35, 0.15, 0.12, 0.10, 0.13, 0.08, 0.07]
    )
    
    home_ownership = np.random.choice(
        ['MORTGAGE', 'RENT', 'OWN', 'OTHER'],
        n_samples, p=[0.45, 0.38, 0.15, 0.02]
    )
    
    # === TARGET VARIABLE: Default (realistic ~15% default rate) ===
    # Default probability driven by realistic risk factors
    log_odds = (
        -2.5
        + 0.8 * (credit_score < 620).astype(float)
        + 0.5 * (credit_score < 680).astype(float)
        - 0.3 * (credit_score > 750).astype(float)
        + 1.2 * (debt_to_income > 0.4).astype(float)
        + 0.6 * (num_delinquencies > 0).astype(float)
        + 0.4 * (credit_utilization > 0.6).astype(float)
        - 0.3 * (employment_length > 5).astype(float)
        + 0.3 * (loan_amount > 30000).astype(float)
        + 0.5 * (interest_rate > 18).astype(float)
        - 0.2 * (annual_income > 80000).astype(float)
        + np.random.normal(0, 0.5, n_samples)  # noise
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    default = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Introduce ~3% missing values in some columns (realistic)
    mask_income = np.random.random(n_samples) < 0.03
    mask_emp = np.random.random(n_samples) < 0.02
    
    df = pd.DataFrame({
        'age': age,
        'annual_income': annual_income.astype(float),
        'employment_length_years': employment_length.astype(float),
        'credit_score': credit_score,
        'num_credit_lines': num_credit_lines,
        'credit_utilization': credit_utilization,
        'num_delinquencies': num_delinquencies,
        'months_since_last_delinquency': months_since_delinquency,
        'loan_amount': loan_amount,
        'loan_term_months': loan_term_months,
        'interest_rate': np.round(interest_rate, 2),
        'debt_to_income_ratio': np.round(debt_to_income, 4),
        'loan_purpose': loan_purpose,
        'home_ownership': home_ownership,
        'default': default
    })
    
    # Add missing values
    df.loc[mask_income, 'annual_income'] = np.nan
    df.loc[mask_emp, 'employment_length_years'] = np.nan
    
    return df


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def run_eda(df, save_dir='outputs'):
    """Comprehensive EDA with publication-quality visualizations."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # --- Dataset Overview ---
    print(f"\nDataset Shape: {df.shape}")
    print(f"Default Rate: {df['default'].mean():.1%}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nFeature Statistics:\n{df.describe().round(2)}")
    
    # --- Figure 1: Default Rate by Key Features ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Loan Default Rate by Key Risk Factors', fontsize=16, fontweight='bold', y=1.02)
    
    # Credit Score Bins
    df['credit_score_bin'] = pd.cut(df['credit_score'], bins=[299, 580, 620, 680, 740, 850],
                                     labels=['Poor\n(300-580)', 'Fair\n(581-620)', 'Good\n(621-680)',
                                             'Very Good\n(681-740)', 'Excellent\n(741-850)'])
    default_by_score = df.groupby('credit_score_bin', observed=True)['default'].mean()
    colors = ['#d32f2f', '#f57c00', '#fbc02d', '#66bb6a', '#2e7d32']
    axes[0, 0].bar(default_by_score.index, default_by_score.values, color=colors, edgecolor='white')
    axes[0, 0].set_title('Default Rate by Credit Score', fontweight='bold')
    axes[0, 0].set_ylabel('Default Rate')
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # DTI Ratio
    df['dti_bin'] = pd.cut(df['debt_to_income_ratio'], bins=[0, 0.15, 0.3, 0.45, 1.0],
                            labels=['Low\n(<15%)', 'Medium\n(15-30%)', 'High\n(30-45%)', 'Very High\n(>45%)'])
    default_by_dti = df.groupby('dti_bin', observed=True)['default'].mean()
    axes[0, 1].bar(default_by_dti.index, default_by_dti.values, color=['#2e7d32', '#fbc02d', '#f57c00', '#d32f2f'], edgecolor='white')
    axes[0, 1].set_title('Default Rate by Debt-to-Income', fontweight='bold')
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Loan Purpose
    default_by_purpose = df.groupby('loan_purpose')['default'].mean().sort_values(ascending=True)
    axes[0, 2].barh(default_by_purpose.index, default_by_purpose.values, color='#1976d2', edgecolor='white')
    axes[0, 2].set_title('Default Rate by Loan Purpose', fontweight='bold')
    axes[0, 2].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Interest Rate
    df['rate_bin'] = pd.cut(df['interest_rate'], bins=[0, 8, 13, 18, 30],
                             labels=['<8%', '8-13%', '13-18%', '>18%'])
    default_by_rate = df.groupby('rate_bin', observed=True)['default'].mean()
    axes[1, 0].bar(default_by_rate.index, default_by_rate.values, color=['#2e7d32', '#fbc02d', '#f57c00', '#d32f2f'], edgecolor='white')
    axes[1, 0].set_title('Default Rate by Interest Rate', fontweight='bold')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Home Ownership
    default_by_home = df.groupby('home_ownership')['default'].mean().sort_values(ascending=True)
    axes[1, 1].barh(default_by_home.index, default_by_home.values, color='#7b1fa2', edgecolor='white')
    axes[1, 1].set_title('Default Rate by Home Ownership', fontweight='bold')
    axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Delinquency History
    df['delinq_flag'] = np.where(df['num_delinquencies'] > 0, 'Has Delinquencies', 'No Delinquencies')
    default_by_delinq = df.groupby('delinq_flag')['default'].mean()
    axes[1, 2].bar(default_by_delinq.index, default_by_delinq.values, color=['#2e7d32', '#d32f2f'], edgecolor='white')
    axes[1, 2].set_title('Default Rate by Delinquency History', fontweight='bold')
    axes[1, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_default_rate_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {save_dir}/01_default_rate_analysis.png")
    
    # --- Figure 2: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 9))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/02_correlation_heatmap.png")
    
    # --- Figure 3: Distribution of Key Features ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Distributions: Default vs Non-Default', fontsize=14, fontweight='bold')
    
    for ax, col, title in zip(
        axes.flat,
        ['credit_score', 'annual_income', 'debt_to_income_ratio', 'interest_rate'],
        ['Credit Score', 'Annual Income', 'Debt-to-Income Ratio', 'Interest Rate (%)']
    ):
        for label, color in [(0, '#2e7d32'), (1, '#d32f2f')]:
            subset = df[df['default'] == label][col].dropna()
            ax.hist(subset, bins=40, alpha=0.6, color=color, 
                    label='No Default' if label == 0 else 'Default', density=True)
        ax.set_title(title, fontweight='bold')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/03_feature_distributions.png")
    
    # Cleanup temp columns
    df.drop(columns=['credit_score_bin', 'dti_bin', 'rate_bin', 'delinq_flag'], inplace=True, errors='ignore')
    
    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Create domain-driven features that capture lending risk signals.
    
    This is where banking domain knowledge shines in interviews —
    employers want to see you can translate business logic into features.
    """
    df = df.copy()
    
    # Income-to-Loan ratio (affordability signal)
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    # Monthly payment estimate
    monthly_rate = df['interest_rate'] / 100 / 12
    n_payments = df['loan_term_months']
    df['est_monthly_payment'] = df['loan_amount'] * (
        monthly_rate * (1 + monthly_rate)**n_payments
    ) / ((1 + monthly_rate)**n_payments - 1)
    
    # Payment burden (monthly payment as % of monthly income)
    df['payment_burden'] = df['est_monthly_payment'] / (df['annual_income'] / 12 + 1)
    
    # Credit risk score (composite)
    df['credit_risk_composite'] = (
        (850 - df['credit_score']) / 550 * 0.4 +
        df['credit_utilization'] * 0.3 +
        (df['num_delinquencies'] > 0).astype(float) * 0.3
    )
    
    # Employment stability flag
    df['stable_employment'] = (df['employment_length_years'] >= 3).astype(int)
    
    # High risk flags (interpretable binary features)
    df['high_dti'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
    df['high_utilization'] = (df['credit_utilization'] > 0.6).astype(int)
    df['subprime'] = (df['credit_score'] < 620).astype(int)
    
    print(f"✓ Engineered {8} new features")
    print(f"  Total features: {df.shape[1] - 1}")
    
    return df


# ============================================================
# 4. PREPROCESSING
# ============================================================

def preprocess(df):
    """Prepare data for modeling: encode categoricals, handle missing values, scale."""
    df = df.copy()
    
    # Encode categoricals
    le_purpose = LabelEncoder()
    le_home = LabelEncoder()
    df['loan_purpose_encoded'] = le_purpose.fit_transform(df['loan_purpose'])
    df['home_ownership_encoded'] = le_home.fit_transform(df['home_ownership'])
    
    # One-hot encode for better model performance
    df = pd.get_dummies(df, columns=['loan_purpose', 'home_ownership'], drop_first=True)
    
    # Define feature set (drop target + intermediate columns)
    drop_cols = ['default', 'loan_purpose_encoded', 'home_ownership_encoded']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['default']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Train/Test Split (stratified to maintain default ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    print(f"✓ Preprocessing complete")
    print(f"  Train set: {X_train_scaled.shape[0]:,} samples")
    print(f"  Test set:  {X_test_scaled.shape[0]:,} samples")
    print(f"  Features:  {X_train_scaled.shape[1]}")
    print(f"  Default rate (train): {y_train.mean():.1%}")
    print(f"  Default rate (test):  {y_test.mean():.1%}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


# ============================================================
# 5. MODEL TRAINING & COMPARISON
# ============================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols, save_dir='outputs'):
    """
    Train multiple models, compare performance, and generate
    interpretability plots that banking hiring managers love.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, C=0.5
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=20, random_state=42, subsample=0.8
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Fit on full training set
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_prob)
        
        print(f"  Test AUC:  {auc:.4f}")
        print(f"  Test F1:   {f1:.4f}")
        print(f"  Test AP:   {ap:.4f}")
        print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Default', 'Default'])}")
        
        results[name] = {
            'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
            'auc': auc, 'f1': f1, 'ap': ap, 'cv_auc': cv_scores.mean()
        }
    
    # === VISUALIZATION: Model Comparison ===
    
    # --- Figure 4: ROC Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'Logistic Regression': '#1976d2', 'Random Forest': '#2e7d32', 'Gradient Boosting': '#d32f2f'}
    
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=colors[name], linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall Curve
    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        axes[1].plot(recall, precision, label=f"{name} (AP={res['ap']:.3f})", color=colors[name], linewidth=2)
    
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/04_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {save_dir}/04_model_comparison.png")
    
    # --- Figure 5: Feature Importance (Gradient Boosting - best model) ---
    best_name = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        top_features = importances.nlargest(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features.sort_values().plot(kind='barh', ax=ax, color='#1976d2', edgecolor='white')
        ax.set_title(f'Top 15 Feature Importances ({best_name})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/05_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_dir}/05_feature_importance.png")
    
    # --- Figure 6: Confusion Matrix (Best Model) ---
    best_res = results[best_name]
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/06_confusion_matrix.png")
    
    # --- Figure 7: Risk Score Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color, name in [(0, '#2e7d32', 'No Default'), (1, '#d32f2f', 'Default')]:
        mask = y_test == label
        ax.hist(best_res['y_prob'][mask], bins=50, alpha=0.6, color=color, label=name, density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax.set_xlabel('Predicted Default Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Risk Score Distribution by Actual Outcome', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/07_risk_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/07_risk_score_distribution.png")
    
    return results, best_name


# ============================================================
# 6. BUSINESS IMPACT ANALYSIS
# ============================================================

def business_impact_analysis(y_test, best_results, save_dir='outputs'):
    """
    Translate model performance into business value.
    THIS is what gets you hired — showing you think like a banker, not just a coder.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("BUSINESS IMPACT ANALYSIS")
    print("=" * 60)
    
    y_prob = best_results['y_prob']
    
    # Simulate financial impact at different thresholds
    avg_loan = 25000  # average loan amount
    loss_given_default = 0.6  # 60% loss if default occurs
    interest_margin = 0.05  # 5% profit margin on good loans
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    profits = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        
        # True positives = correctly caught defaults (saved losses)
        tp = ((y_pred_t == 1) & (y_test == 1)).sum()
        # False positives = good borrowers rejected (lost revenue)
        fp = ((y_pred_t == 1) & (y_test == 0)).sum()
        # False negatives = missed defaults (actual losses)
        fn = ((y_pred_t == 0) & (y_test == 1)).sum()
        # True negatives = correctly approved good loans (revenue)
        tn = ((y_pred_t == 0) & (y_test == 0)).sum()
        
        saved_losses = tp * avg_loan * loss_given_default
        lost_revenue = fp * avg_loan * interest_margin
        actual_losses = fn * avg_loan * loss_given_default
        earned_revenue = tn * avg_loan * interest_margin
        
        net_value = saved_losses + earned_revenue - lost_revenue - actual_losses
        profits.append({
            'threshold': t, 'net_value': net_value,
            'defaults_caught': tp, 'good_loans_rejected': fp,
            'defaults_missed': fn, 'good_loans_approved': tn
        })
    
    profit_df = pd.DataFrame(profits)
    optimal_idx = profit_df['net_value'].idxmax()
    optimal = profit_df.iloc[optimal_idx]
    
    print(f"\n  Optimal Threshold: {optimal['threshold']:.2f}")
    print(f"  Net Value at Optimal: ${optimal['net_value']:,.0f}")
    print(f"  Defaults Caught: {optimal['defaults_caught']:.0f}")
    print(f"  Good Loans Rejected: {optimal['good_loans_rejected']:.0f}")
    print(f"  Defaults Missed: {optimal['defaults_missed']:.0f}")
    
    # Scale to portfolio level
    portfolio_multiplier = 100  # scale to 200K loans
    annual_savings = (optimal['net_value'] - profit_df.iloc[0]['net_value']) * portfolio_multiplier
    print(f"\n  Estimated Annual Savings (200K loan portfolio): ${annual_savings:,.0f}")
    
    # --- Figure 8: Profit Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(profit_df['threshold'], profit_df['net_value'] / 1e6, color='#1976d2', linewidth=2.5)
    ax.axvline(x=optimal['threshold'], color='#d32f2f', linestyle='--', alpha=0.7,
               label=f"Optimal Threshold: {optimal['threshold']:.2f}")
    ax.scatter([optimal['threshold']], [optimal['net_value'] / 1e6], color='#d32f2f', s=100, zorder=5)
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Net Value ($M)', fontsize=12)
    ax.set_title('Profit Optimization: Finding the Best Decision Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/08_profit_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {save_dir}/08_profit_curve.png")
    
    return optimal


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║   LOAN DEFAULT RISK PREDICTION — PORTFOLIO PROJECT       ║")
    print("╚" + "═" * 58 + "╝")
    
    # Step 1: Generate Data
    print("\n[1/5] Generating loan dataset...")
    df = generate_loan_data(n_samples=10000)
    
    # Step 2: EDA
    print("\n[2/5] Running Exploratory Data Analysis...")
    df = run_eda(df)
    
    # Step 3: Feature Engineering
    print("\n[3/5] Engineering features...")
    df = engineer_features(df)
    
    # Step 4: Preprocessing
    print("\n[4/5] Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)
    
    # Step 5: Train & Evaluate Models
    print("\n[5/5] Training and evaluating models...")
    results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols)
    
    # Step 6: Business Impact
    optimal = business_impact_analysis(y_test, results[best_name])
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Best Model: {best_name}")
    print(f"  Test AUC:   {results[best_name]['auc']:.4f}")
    print(f"  Test F1:    {results[best_name]['f1']:.4f}")
    print(f"\n  All outputs saved to: outputs/")
    print(f"\n  Next Steps:")
    print(f"  → Upload to GitHub with the README.md")
    print(f"  → Link in your resume under 'Projects'")
    print(f"  → Be ready to discuss business impact in interviews")
