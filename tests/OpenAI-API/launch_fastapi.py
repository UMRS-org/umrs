#!/usr/bin/env python3

# launch_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import uvicorn

# ----- Créer le serveur FastAPI -----
app = FastAPI(title="OpenAI-compatible Embedding Server")

# ----- Charger le modèle d'embeddings -----
MODEL_NAME = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # Mode évaluation, pas de calculs de gradients


# ----- Définir les classes de requêtes et réponses -----
class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: List[dict]


# ----- Endpoint pour calculer les embeddings -----
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    if req.model != MODEL_NAME:
        raise ValueError(f"Unsupported model: {req.model}")

    # Tokenizer les textes
    inputs = tokenizer(req.input, padding=True, truncation=True, return_tensors="pt")

    # Calculer les embeddings (moyenne sur les tokens)
    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state.mean(dim=1).tolist()

    # Retourner sous forme compatible OpenAI
    return {"data": [{"embedding": emb} for emb in embeddings]}


# ----- Endpoint pour lister les modèles disponibles -----
@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_NAME, "object": "model", "max_model_len": 512}]}


# ----- Lancer le serveur -----
if __name__ == "__main__":
    uvicorn.run("launch_fastapi:app", host="0.0.0.0", port=8001, log_level="info")
