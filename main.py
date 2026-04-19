from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import joblib, numpy as np, pandas as pd, os

# Must be defined before joblib.load so pickle can resolve the class
class LightweightDataSanitizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_config):
        self.feature_config = feature_config
        self.numeric_means_ = {}

    def fit(self, X, y=None):
        for col in self.feature_config['numeric_features']:
            if col in X.columns:
                self.numeric_means_[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X = X.rename(columns=lambda x: x.strip().lower())
        expected = self.feature_config['expected_features']
        for col in expected:
            if col not in X.columns:
                X[col] = self.numeric_means_.get(col, 0) if col in self.feature_config['numeric_features'] else 0
        X = X[expected]
        for col in expected:
            if col in self.feature_config['numeric_features']:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.numeric_means_.get(col, 0))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
        return X


app = FastAPI(title="Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "optimized_credit_model.pkl")
bundle = joblib.load(MODEL_PATH)
pipeline  = bundle["deployment_pipeline"]
threshold = bundle.get("threshold", 0.5)


class LoanApplication(BaseModel):
    Age: int
    Income: float               # annual
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int               # months
    DTIRatio: float             # 0–1
    Education: str              # HighSchool | Bachelor | Master | PhD
    EmploymentType: str         # Full-time | Part-time | Self-employed | Unemployed
    MaritalStatus: str          # Single | Married | Divorced
    HasMortgage: int            # 0 | 1
    HasDependents: int          # 0 | 1
    LoanPurpose: str            # Home | Auto | Education | Business | Other
    HasCoSigner: int            # 0 | 1


EDUCATION_MAP  = {"HighSchool": "High School", "Bachelor": "Bachelor's", "Master": "Master's", "PhD": "PhD"}
EMPLOYMENT_MAP = {"Full-time": "Full-time", "Part-time": "Part-time", "Self-employed": "Self-employed", "Unemployed": "Unemployed"}


@app.post("/predict")
def predict(app_in: LoanApplication):
    try:
        row = {
            "Age":            app_in.Age,
            "Income":         app_in.Income,
            "LoanAmount":     app_in.LoanAmount,
            "CreditScore":    app_in.CreditScore,
            "MonthsEmployed": app_in.MonthsEmployed,
            "NumCreditLines": app_in.NumCreditLines,
            "InterestRate":   app_in.InterestRate,
            "LoanTerm":       app_in.LoanTerm,
            "DTIRatio":       app_in.DTIRatio,
            "Education":      EDUCATION_MAP.get(app_in.Education, app_in.Education),
            "EmploymentType": EMPLOYMENT_MAP.get(app_in.EmploymentType, app_in.EmploymentType),
            "MaritalStatus":  app_in.MaritalStatus,
            "HasMortgage":    app_in.HasMortgage,
            "HasDependents":  app_in.HasDependents,
            "LoanPurpose":    app_in.LoanPurpose,
            "HasCoSigner":    app_in.HasCoSigner,
        }
        df = pd.DataFrame([row])
        prob = float(pipeline.predict_proba(df)[0, 1])
        decision = "Decline" if prob > threshold else ("Review" if prob > threshold * 0.7 else "Approve")

        if prob >= 0.7:
            risk_level = "High"
        elif prob >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "default_probability": round(prob, 4),
            "threshold": threshold,
            "decision": decision,
            "risk_level": risk_level,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": "optimized_credit_model", "threshold": threshold}
