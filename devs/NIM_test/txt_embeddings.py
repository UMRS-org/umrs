#!/usr/bin/python3

from transformers import AutoTokenizer, AutoModel
import torch

# Load E5 model from Hugging Face
model_name = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample text
text = "This is an example sentence for embedding."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Compute embedding
with torch.no_grad():
    outputs = model(**inputs)

# Mean pooling
embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
print("Embedding Vector:", embeddings)
print("Len embedding Vector:", len(embeddings))
