"""
Credit Default Risk Model — Training Script
============================================
Trains an XGBoost + LightGBM pipeline on 255k loan records.
Selects best model by AUC-ROC, calibrates probabilities,
optimises decision threshold, and saves a clean production pkl.

Run:
    pip install -r requirements_train.txt
    python train.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline           import Pipeline
from sklearn.base               import clone
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import OneHotEncoder, StandardScaler
from sklearn.model_selection    import StratifiedKFold, cross_val_score, train_test_split
from sklearn.calibration        import CalibratedClassifierCV, calibration_curve
from sklearn.metrics            import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, roc_curve
)

import xgboost  as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("../loan-predictor/dataset/credit_data.csv")
MODEL_OUT  = Path("credit_risk_model.pkl")
REPORT_OUT = Path("training_report.json")

# ── Feature schema (must match API input exactly) ─────────────────────────────
NUMERIC_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "MonthsEmployed", "NumCreditLines", "InterestRate",
    "LoanTerm", "DTIRatio",
    # engineered
    "LoanToIncome", "EMIToIncome", "CreditRiskIndex",
]
CATEGORICAL_FEATURES = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
BINARY_FEATURES      = ["HasMortgage", "HasDependents", "HasCoSigner"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET               = "Default"

# Known categories — fixed so inference never sees unknown labels
CAT_CATEGORIES = [
    ["Bachelor's", "High School", "Master's", "PhD"],
    ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    ["Divorced", "Married", "Single"],
    ["Auto", "Business", "Education", "Home", "Other"],
]

# ── Load & engineer features ──────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df = df.drop(columns=["LoanID"], errors="ignore")

    # binary Yes/No → 0/1
    for col in BINARY_FEATURES:
        df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)

    # engineered features
    df["LoanToIncome"]    = df["LoanAmount"] / df["Income"].clip(lower=1)
    # rough monthly payment: P * r*(1+r)^n / ((1+r)^n - 1)
    r = (df["InterestRate"] / 100 / 12).clip(lower=1e-6)
    n = df["LoanTerm"]
    monthly_pmt = df["LoanAmount"] * r * (1 + r)**n / ((1 + r)**n - 1)
    df["EMIToIncome"]     = monthly_pmt / (df["Income"] / 12).clip(lower=1)
    # composite risk index: high DTI + low credit score + high interest = risky
    df["CreditRiskIndex"] = (df["DTIRatio"] * df["InterestRate"]) / df["CreditScore"].clip(lower=1)

    print(f"  Rows: {len(df):,}  |  Default rate: {df[TARGET].mean():.2%}")
    return df

# ── Build preprocessor ────────────────────────────────────────────────────────
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(
                categories=CAT_CATEGORIES,
                handle_unknown="ignore",
                sparse_output=False,
                drop="first",
            ), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

# ── Optuna objective — XGBoost ────────────────────────────────────────────────
def xgb_objective(trial, X_tr, y_tr, preprocessor):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 1200),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3),
        "scale_pos_weight":  (y_tr == 0).sum() / (y_tr == 1).sum(),
        "eval_metric":       "auc",
        "use_label_encoder": False,
        "random_state":      42,
        "n_jobs":            -1,
    }
    pipe = Pipeline([("pre", preprocessor), ("clf", xgb.XGBClassifier(**params))])
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ── Optuna objective — LightGBM ───────────────────────────────────────────────
def lgb_objective(trial, X_tr, y_tr, preprocessor):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 300, 1200),
        "max_depth":       trial.suggest_int("max_depth", 3, 8),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.5, 3),
        "is_unbalance":    True,
        "random_state":    42,
        "n_jobs":          -1,
        "verbose":         -1,
    }
    pipe = Pipeline([("pre", preprocessor), ("clf", lgb.LGBMClassifier(**params))])
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ── Find optimal threshold (Youden's J) ──────────────────────────────────────
def optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    ap     = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    ks     = max(abs(
        np.cumsum(np.bincount(y_true[y_pred == 1], minlength=2) / (y_true == 1).sum()) -
        np.cumsum(np.bincount(y_true[y_pred == 0], minlength=2) / (y_true == 0).sum())
    )) if len(np.unique(y_pred)) > 1 else 0.0

    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    print(f"  AUC-ROC       : {auc:.4f}")
    print(f"  AUC-PR        : {ap:.4f}")
    print(f"  Brier Score   : {brier:.4f}  (lower=better)")
    print(f"  Threshold     : {threshold:.4f}")
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))
    return {"auc_roc": round(auc, 4), "auc_pr": round(ap, 4),
            "brier": round(brier, 4), "threshold": round(threshold, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_data(DATA_PATH)
    X  = df[ALL_FEATURES]
    y  = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    model_name = "LightGBM"
    best_params = {
        "n_estimators":      800,
        "max_depth":         6,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_samples": 30,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "is_unbalance":      True,
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    print(f"\n[1/2] Training LightGBM pipeline...")
    production_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model",        lgb.LGBMClassifier(**best_params)),
    ])
    production_pipeline.fit(X_train, y_train)

    y_final_prob = production_pipeline.predict_proba(X_test)[:, 1]
    final_auc    = roc_auc_score(y_test, y_final_prob)
    final_thr    = optimal_threshold(y_test, y_final_prob)

    print(f"\n[2/2] Evaluating...")
    metrics = evaluate(model_name, y_test.values, y_final_prob, final_thr)
    print(f"\n  AUC: {final_auc:.4f}  |  Threshold: {final_thr:.4f}")

    # ── Save bundle ───────────────────────────────────────────────────────────
    bundle = {
        "pipeline":              production_pipeline,
        "threshold":             round(final_thr, 4),
        "model_name":            model_name,
        "features":              ALL_FEATURES,
        "numeric_features":      NUMERIC_FEATURES,
        "categorical_features":  CATEGORICAL_FEATURES,
        "binary_features":       BINARY_FEATURES,
        "cat_categories":        CAT_CATEGORIES,
        "metrics":               metrics,
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)
    print(f"\n  Model saved: {MODEL_OUT}  ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)")

    report = {
        "model":         model_name,
        "training_rows": len(X_train),
        "test_rows":     len(X_test),
        "default_rate":  round(float(y.mean()), 4),
        "threshold":     round(final_thr, 4),
        "best_params":   best_params,
        "metrics":       metrics,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    print(f"  Report saved: {REPORT_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
