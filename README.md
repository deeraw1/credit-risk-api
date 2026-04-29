# Credit Risk API

FastAPI backend serving an XGBoost credit default prediction model for microfinance lending. Accepts loan application features and returns default probability, credit grade, and feature importance scores.

## Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/predict` | Score a loan application |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata and performance metrics |

## Request Schema

```json
{
  "Age": 32,
  "Income": 4500000,
  "LoanAmount": 1000000,
  "CreditScore": 620,
  "MonthsEmployed": 24,
  "NumCreditLines": 3,
  "InterestRate": 18.5,
  "LoanTerm": 36,
  "DTIRatio": 0.35,
  "Education": "Bachelor",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": false,
  "HasDependents": true,
  "LoanPurpose": "Business",
  "HasCoSigner": false
}
```

## Response Schema

```json
{
  "default_probability": 0.23,
  "credit_grade": "C",
  "decision": "Refer to underwriter",
  "model": "XGBoost",
  "threshold": 0.42
}
```

## Model

- **XGBoost** classifier trained on synthetic microfinance lending data
- Threshold optimised on F1 score from `train.py`
- Model artefact: `credit_risk_model.pkl` and `optimized_credit_model.pkl`
- Training report: `training_report.json`

## Tech Stack

- **Python 3.11**
- **FastAPI** — REST API framework
- **XGBoost** — gradient boosted classifier
- **scikit-learn** — preprocessing pipeline
- **joblib** — model serialisation
- **Render** — deployment (`render.yaml` included)

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## Retrain Model

```bash
pip install -r requirements_train.txt
python train.py
```

---

Built by [Muhammed Adediran](https://adediran.xyz/contact)
