import torch
from sentence_transformers import SentenceTransformer

from umrs.knowledge.text_embedder import bundled_dir_for
from umrs.shared.bounded_load import load_bounded
from umrs.shared.utils import (
    add_document_prefix,
    add_embeddings_to_chunks_dict,
    flatten_text_chunks,
)

################################ Sentence Transformers ###########################
#  Sentence Transformer Loader
_model_cache = {}


def load_sentence_transformers_model(model_name):
    if model_name in _model_cache:
        print(f"\t\t\t--> Model '{model_name}' already loaded...")
        return _model_cache[model_name]

    print(f"\t\t\t--> Loading model '{model_name}'...")
    # Issue #635: bounded (per-attempt deadline, retry with backoff, cooldown)
    # so a stalled Hugging Face transfer cannot hold boot provisioning past
    # the compose health window.
    model = load_bounded(model_name, lambda: _load_sentence_transformer(model_name))
    _model_cache[model_name] = model
    return model


def _load_sentence_transformer(model_name):
    """One attempt: the image's bundled copy or the local cache, else download."""
    source = bundled_dir_for(model_name) or model_name
    try:
        return SentenceTransformer(source, local_files_only=True)
    except Exception:
        print(
            "\t\t\t--> Local cache not available or incomplete. Downloading from Hugging Face..."
        )
        model = SentenceTransformer(source)
        print("\t\t\t--> Downloaded and cached model.")
        return model


#  Compute Embeddings for Text
def compute_embeddings_text(self, texts):
    flat_texts, chunks_dict = flatten_text_chunks(
        texts, self.nlp, tokenizer=self.tokenizer, max_tokens=self.max_tokens
    )
    flat_texts = add_document_prefix(flat_texts, self.model_name)
    embeddings = self.model.encode(
        flat_texts, convert_to_tensor=(self.device == "cuda")
    )
    embeddings = embeddings.cpu().tolist() if self.device == "cuda" else embeddings
    return add_embeddings_to_chunks_dict(embeddings, chunks_dict)


#  Compute Embeddings for Images
def compute_embeddings_img(self, images):
    # Ensure images are in RGB format
    images = [img.convert("RGB") for img in images]

    # Compute embeddings
    with torch.no_grad():
        embeddings = self.model.encode(
            images, convert_to_tensor=(self.device == "cuda")
        )

    return embeddings.cpu().tolist() if self.device == "cuda" else embeddings


def sentence_transformers(modality):
    if modality in ("text", "text_file"):
        compute_embeddings_func = compute_embeddings_text
        input_type = "text"
    elif modality == "image":
        compute_embeddings_func = compute_embeddings_img
        input_type = "image"
    else:
        raise ValueError(
            f"Unsupported modality '{modality}' for SentenceTransformer model"
        )
    return compute_embeddings_func, input_type
