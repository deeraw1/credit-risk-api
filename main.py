from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, numpy as np, pandas as pd, os
from pathlib import Path

app = FastAPI(title="Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(os.path.dirname(__file__)) / "credit_risk_model.pkl"
bundle    = joblib.load(MODEL_PATH)
pipeline  = bundle["pipeline"]
threshold = bundle["threshold"]
metrics   = bundle.get("metrics", {})
model_name = bundle.get("model_name", "XGBoost")


class LoanApplication(BaseModel):
    Age:            int   = Field(..., ge=18, le=80)
    Income:         float = Field(..., gt=0)            # annual
    LoanAmount:     float = Field(..., gt=0)
    CreditScore:    int   = Field(..., ge=300, le=850)
    MonthsEmployed: int   = Field(..., ge=0)
    NumCreditLines: int   = Field(..., ge=0)
    InterestRate:   float = Field(..., ge=0)
    LoanTerm:       int   = Field(..., ge=1)            # months
    DTIRatio:       float = Field(..., ge=0, le=1)
    Education:      str                                  # Bachelor's | High School | Master's | PhD
    EmploymentType: str                                  # Full-time | Part-time | Self-employed | Unemployed
    MaritalStatus:  str                                  # Divorced | Married | Single
    HasMortgage:    int   = Field(..., ge=0, le=1)
    HasDependents:  int   = Field(..., ge=0, le=1)
    LoanPurpose:    str                                  # Auto | Business | Education | Home | Other
    HasCoSigner:    int   = Field(..., ge=0, le=1)


def engineer_features(d: dict) -> dict:
    income   = max(d["Income"], 1)
    r        = (d["InterestRate"] / 100 / 12) or 1e-6
    n        = d["LoanTerm"]
    monthly_pmt = d["LoanAmount"] * r * (1 + r)**n / ((1 + r)**n - 1)
    d["LoanToIncome"]    = d["LoanAmount"] / income
    d["EMIToIncome"]     = monthly_pmt / (income / 12)
    d["CreditRiskIndex"] = (d["DTIRatio"] * d["InterestRate"]) / max(d["CreditScore"], 1)
    return d


@app.post("/predict")
def predict(app_in: LoanApplication):
    try:
        row = engineer_features(app_in.model_dump())
        df  = pd.DataFrame([row])

        prob     = float(pipeline.predict_proba(df)[0, 1])
        decision = "Decline" if prob >= threshold else ("Review" if prob >= threshold * 0.65 else "Approve")
        risk     = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")

        return {
            "default_probability": round(prob, 4),
            "threshold":           threshold,
            "decision":            decision,
            "risk_level":          risk,
            "model":               model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      model_name,
        "threshold":  threshold,
        "auc_roc":    metrics.get("auc_roc"),
    }
