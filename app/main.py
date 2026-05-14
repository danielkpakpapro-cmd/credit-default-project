"""
API FastAPI - Prédiction Défaut de Prêt Bancaire
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ─── Chargement du modèle ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

model        = joblib.load(MODEL_PATH)
scaler       = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
model_name   = joblib.load(os.path.join(BASE_DIR, "models", "best_model_name.pkl"))

# ─── App FastAPI ─────────────────────────────────────────────
app = FastAPI(
    title="🏦 API - Prédiction Défaut de Prêt",
    description=f"""
API de prédiction du risque de défaut de prêt bancaire.
- **Modèle utilisé** : {model_name}
- **Features** : {', '.join(feature_names)}
""",
    version="1.0.0",
)

# ─── Schéma d'entrée ─────────────────────────────────────────
class ClientData(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Âge du client", example=35)
    Experience: int = Field(..., ge=0, le=50, description="Années d'expérience", example=10)
    Income: int = Field(..., ge=0, description="Revenu annuel (en milliers $)", example=80)
    Family: int = Field(..., ge=1, le=4, description="Taille de la famille (1-4)", example=3)
    CCAvg: float = Field(..., ge=0, description="Dépenses mensuelles carte de crédit (milliers $)", example=2.5)
    Education: int = Field(..., ge=1, le=3, description="Niveau d'éducation (1=Lycée, 2=Licence, 3=Master+)", example=2)
    Mortgage: int = Field(..., ge=0, description="Montant du prêt immobilier (milliers $)", example=0)
    Securities_Account: int = Field(..., ge=0, le=1, description="Compte titres (0=Non, 1=Oui)", example=0)
    CD_Account: int = Field(..., ge=0, le=1, description="Compte dépôt à terme (0=Non, 1=Oui)", example=0)
    Online: int = Field(..., ge=0, le=1, description="Banque en ligne (0=Non, 1=Oui)", example=1)
    CreditCard: int = Field(..., ge=0, le=1, description="Carte de crédit banque (0=Non, 1=Oui)", example=0)

# ─── Schéma de réponse ───────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_default: float
    probability_no_default: float
    risk_level: str
    model_used: str

# ─── Routes ──────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "message": "🏦 API Prédiction Défaut de Prêt Bancaire",
        "model": model_name,
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health":  "/health",
            "docs":    "/docs",
        }
    }

@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "model": model_name}

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(client: ClientData):
    try:
        # Construire le vecteur de features dans le bon ordre
        input_dict = {
            "Age": client.Age,
            "Experience": client.Experience,
            "Income": client.Income,
            "Family": client.Family,
            "CCAvg": client.CCAvg,
            "Education": client.Education,
            "Mortgage": client.Mortgage,
            "Securities Account": client.Securities_Account,
            "CD Account": client.CD_Account,
            "Online": client.Online,
            "CreditCard": client.CreditCard,
        }
        features = np.array([[input_dict[f] for f in feature_names]])

        # Normalisation
        features_scaled = scaler.transform(features)

        # Prédiction
        pred = int(model.predict(features_scaled)[0])
        proba = model.predict_proba(features_scaled)[0]
        prob_default = float(proba[1])
        prob_no_default = float(proba[0])

        # Niveau de risque
        if prob_default < 0.2:
            risk_level = "🟢 Faible"
        elif prob_default < 0.5:
            risk_level = "🟡 Modéré"
        elif prob_default < 0.75:
            risk_level = "🟠 Élevé"
        else:
            risk_level = "🔴 Très élevé"

        return PredictionResponse(
            prediction=pred,
            prediction_label="Défaut probable" if pred == 1 else "Pas de défaut",
            probability_default=round(prob_default, 4),
            probability_no_default=round(prob_no_default, 4),
            risk_level=risk_level,
            model_used=model_name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prédiction"])
def predict_batch(clients: list[ClientData]):
    """Prédiction pour plusieurs clients à la fois"""
    results = []
    for client in clients:
        result = predict(client)
        results.append(result)
    return {"predictions": results, "total": len(results)}
