# api/main.py
# API FastAPI pour SenSante - Assistant pré-diagnostic médical

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Client Groq (chargé au démarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvée. /explain sera désactivé.")


# --- Schemas Pydantic ---

class PatientInput(BaseModel):
    """Données d'entrée : les symptômes d'un patient."""
    age: int = Field(..., ge=0, le=120, description="Âge en années")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(
        ..., ge=35.0, le=42.0,
        description="Température en Celsius"
    )
    tension_sys: int = Field(
        ..., ge=60, le=250,
        description="Tension systolique"
    )
    toux: bool = Field(..., description="Présence de toux")
    fatigue: bool = Field(..., description="Présence de fatigue")
    maux_tete: bool = Field(..., description="Présence de maux de tête")
    region: str = Field(..., description="Région du Sénégal")


class DiagnosticOutput(BaseModel):
    """Données de sortie : le résultat du diagnostic."""
    diagnostic: str = Field(..., description="Diagnostic prédit")
    probabilite: float = Field(..., description="Probabilité du diagnostic")
    confiance: str = Field(..., description="Niveau de confiance")
    message: str = Field(..., description="Recommandation")



# Créer l'application
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

# Configurer CORS (utile si l'API est consommée par une app web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ à restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import joblib
import numpy as np

# --- Charger le modèle au démarrage ---
print("Chargement du modèle...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modèle chargé : {list(model.classes_)}")


# Définir un modèle de réponse avec Pydantic
class HealthResponse(BaseModel):
    status: str
    message: str

# Route de base : vérifier que l'API fonctionne
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Vérification de l'état de l'API."""
    return HealthResponse(
        status="ok",
        message="SenSante API is running"
    )

@app.post("/predict", response_model=DiagnosticOutput, tags=["Diagnostic"])
def predict(patient: PatientInput):
    """Prédire un diagnostic à partir des symptômes."""
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
            confiance="aucune", message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F.")
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(diagnostic="erreur", probabilite=0.0,
            confiance="aucune", message=f"Région inconnue : {patient.region}")

    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete), region_enc
    ]])

    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())
    confiance = "haute" if proba_max >= 0.7 else "moyenne" if proba_max >= 0.4 else "faible"

    messages = {
        "palu": "Suspicion de paludisme. Consultez un médecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandés.",
        "typh": "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain": "Pas de pathologie détectée. Continuez à surveiller."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )
# --- Schemas pour /explain ---

class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic prédit par le modèle")
    probabilite: float = Field(..., description="Probabilité du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en français")
    modele_llm: str = Field(default="llama-3.1-8b-instant")

SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""

@app.post("/explain", response_model=ExplainOutput, tags=["LLM"])
def explain(data: ExplainInput):
    """Expliquer un diagnostic en français avec Llama 3."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic du modèle : {data.diagnostic} (probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)
