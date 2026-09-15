#!/usr/bin/python3
import sys
import time

import spacy
from charset_normalizer import from_path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from utils import compute_md5


def chunk_text_with_offsets(text: str, tokenizer, max_tokens: int, nlp) -> list[dict]:
    """
    Split a large text into smaller chunks based on sentences and token count,
    while preserving the exact character offsets (start/end) of each chunk
    relative to the original text.

    Parameters:
    - text: the full input text (already decoded as string)
    - tokenizer: a HuggingFace tokenizer compatible with the embedding model
    - max_tokens: maximum number of tokens per chunk
    - nlp: a spaCy language model for sentence segmentation

    Returns:
    - A list of dictionaries, each containing:
        - text: the chunk text
        - start: start character position in the original text
        - end: end character position in the original text
        - length: character length of the chunk
        - md5: md5 checksum of the chunk
    """
    doc = nlp(text)
    chunks = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        # Tokenize the sentence and get the mapping between tokens and character offsets
        encoding = tokenizer(
            sentence_text, return_offsets_mapping=True, add_special_tokens=False
        )
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        if len(input_ids) <= max_tokens:
            # If the sentence fits in the limit, use it directly
            start = sent.start_char
            end = sent.end_char
            chunks.append(
                {
                    "text": sentence_text,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "md5": compute_md5(sentence_text),
                }
            )
        else:
            # Otherwise, split the sentence into subchunks of max_tokens
            for i in range(0, len(input_ids), max_tokens):
                input_ids[i : i + max_tokens]
                sub_offsets = offsets[i : i + max_tokens]

                # Get the character range covered by the token span
                sub_start = sub_offsets[0][0]
                sub_end = sub_offsets[-1][1]

                # Extract the substring using the token offset range
                sub_text = sentence_text[sub_start:sub_end].strip()
                global_start = sent.start_char + sub_start
                global_end = sent.start_char + sub_end

                if sub_text:
                    chunks.append(
                        {
                            "text": sub_text,
                            "start": global_start,
                            "end": global_end,
                            "length": global_end - global_start,
                            "md5": compute_md5(sub_text),
                        }
                    )

    return chunks


def chunk_text_by_max_tokens(doc, tokenizer, max_tokens: int) -> list[dict]:
    """
    Split a large text into smaller chunks based on sentences and token count,
    while preserving the exact character offsets (start/end) of each chunk
    relative to the original text.

    Parameters:
    - doc: a text segmented by spaCy using 'nlp' function
    - tokenizer: a HuggingFace tokenizer compatible with the embedding model
    - max_tokens: maximum number of tokens per chunk

    Returns:
    - A list of dictionaries, each containing:
        - text: the chunk text
        - start: start character position in the original text
        - end: end character position in the original text
        - length: character length of the chunk
        - md5: md5 checksum of the chunk
    """
    chunks = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        # Tokenize the sentence and get the mapping between tokens and character offsets
        encoding = tokenizer(
            sentence_text, return_offsets_mapping=True, add_special_tokens=False
        )
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        if len(input_ids) <= max_tokens:
            # If the sentence fits in the limit, use it directly
            start = sent.start_char
            end = sent.end_char
            chunks.append(
                {
                    "text": sentence_text,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "md5": compute_md5(sentence_text),
                }
            )
        else:
            # Otherwise, split the sentence into subchunks of max_tokens
            print("Splitings...")
            for i in range(0, len(input_ids), max_tokens):
                input_ids[i : i + max_tokens]
                sub_offsets = offsets[i : i + max_tokens]

                # Get the character range covered by the token span
                sub_start = sub_offsets[0][0]
                sub_end = sub_offsets[-1][1]

                # Extract the substring using the token offset range
                sub_text = sentence_text[sub_start:sub_end].strip()
                global_start = sent.start_char + sub_start
                global_end = sent.start_char + sub_end

                if sub_text:
                    chunks.append(
                        {
                            "text": sub_text,
                            "start": global_start,
                            "end": global_end,
                            "length": global_end - global_start,
                            "md5": compute_md5(sub_text),
                        }
                    )

    return chunks


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
