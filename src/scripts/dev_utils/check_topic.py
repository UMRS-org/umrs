#!/usr/bin/python3
import sys
import time

import spacy
from charset_normalizer import from_path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Canonical chunking lives in umrs.shared.utils (drift R6.2, 2026-08-23):
# the two previous local reimplementations were byte-for-byte near-dupes.
from umrs.shared.utils import chunk_text_by_max_tokens as _canonical_chunk_by_max_tokens
from umrs.shared.utils import compute_md5


def _with_length(chunks: list[dict]) -> list[dict]:
    """Preserve this script's historic ``length`` field (= end - start)."""
    for chunk in chunks:
        chunk["length"] = chunk["end"] - chunk["start"]
    return chunks


def chunk_text_with_offsets(text: str, tokenizer, max_tokens: int, nlp) -> list[dict]:
    """Canonical chunking with this script's per-chunk ``length`` field."""
    return _with_length(
        _canonical_chunk_by_max_tokens(nlp(text), tokenizer, max_tokens)
    )


def chunk_text_by_max_tokens(doc, tokenizer, max_tokens: int) -> list[dict]:
    """Canonical chunking with this script's per-chunk ``length`` field."""
    return _with_length(_canonical_chunk_by_max_tokens(doc, tokenizer, max_tokens))


# Modèle utilisé
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_TOKENS = 2048  # max tokens pour ce modèle (souvent 512 ou 1024 selon le modèle)

# Chargement spaCy et SentenceTransformer
start = time.time()
# nlp = spacy.load("en_core_web_sm")  # anglais, sinon utiliser xx_ent_wiki_sm pour multilingue
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
nlp.add_pipe("sentencizer")  # ajoute un séparateur de phrases simple
print(f"[Timing] spaCy load: {time.time() - start:.2f}s")
start = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"[Timing] model load: {time.time() - start:.2f}s")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"[Timing] Tokenizer load: {time.time() - start:.2f}s")


# Vérification des arguments
if len(sys.argv) != 2:
    print("Usage: python script.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]
# Chrono global
t0 = time.time()

# Étape 1 : Détection et lecture du fichier
start = time.time()
match = from_path(file_path).best()
if match is None:
    raise ValueError("Could not detect encoding of the file.")
text = str(match)
print(f"[Timing] File reading: {time.time() - start:.2f}s")

# Étape 2 : Analyse linguistique avec spaCy
start = time.time()
doc = nlp(text)
print(f"[Timing] spaCy processing: {time.time() - start:.2f}s")

# Étape 3 : Découpage en chunks et calcul des métadonnées
start = time.time()
chunks = chunk_text_by_max_tokens(doc, tokenizer, MAX_TOKENS)
print(f"[Timing] Chunking and MD5 ({len(chunks)} chunks): {time.time() - start:.2f}s")

print(chunks[0])
print(chunks[1])
# sys.exit()

# Étape 4 : Encodage des embeddings
start = time.time()
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, convert_to_tensor=True)
print(f"[Timing] Embedding computation: {time.time() - start:.2f}s")

# Étape 5 : Ajout des embeddings dans chaque chunk
start = time.time()
for chunk, embedding in zip(chunks, embeddings, strict=False):
    chunk["embedding"] = embedding
print(f"[Timing] Embedding attachment to chunks: {time.time() - start:.2f}s")

print(f"[Total Time] {time.time() - t0:.2f}s")
print(f"[Embedding Shape] {embeddings.shape}")

# Assuming you have:
# - `chunks`: list of dictionaries with "start", "end", "md5"
# - `text`: the original full string

nb_errors = 0
for idx, chunk in enumerate(chunks):
    extracted_text = text[chunk["start"] : chunk["end"]].strip()
    computed_md5 = compute_md5(extracted_text)
    if computed_md5 != chunk["md5"]:
        print(f"[Mismatch] Chunk {idx}: MD5 mismatch.")
        print(f"  Expected: {chunk['md5']}")
        print(f"  Found:    {computed_md5}")
        print(f"  Extracted text: {repr(extracted_text)}")
        print(f"  Original chunk text: {repr(chunk['text'])}")
        nb_errors += 1

print(f"MD5 check error: {nb_errors}")
