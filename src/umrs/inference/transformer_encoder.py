import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging

from umrs.knowledge.bundled_embedder import bundled_dir_for
from umrs.shared.bounded_load import load_bounded
from umrs.shared.utils import add_embeddings_to_chunks_dict, flatten_text_chunks

################################ TransformerEncoder  ###########################

_model_cache = {}


def load_transformer_encoder(model_name):
    if model_name in _model_cache:
        print(f"\t\t\t--> Model '{model_name}' already loaded...")
        return _model_cache[model_name]

    logging.set_verbosity_error()  # avoid verbose warning
    print(f"\t\t\t--> Loading model '{model_name}'...")
    # Issue #635: bounded like the SentenceTransformer loader.
    model, tokenizer = load_bounded(model_name, lambda: _load_encoder(model_name))
    model.eval()
    _model_cache[model_name] = (model, tokenizer)
    return model, tokenizer


def _from_pretrained(source, **kwargs):
    return (
        AutoModel.from_pretrained(source, **kwargs),
        AutoTokenizer.from_pretrained(source, **kwargs),
    )


def _load_encoder(model_name):
    """One attempt: the image's bundled copy or the local cache, else download."""
    source = bundled_dir_for(model_name) or model_name
    try:
        return _from_pretrained(source, local_files_only=True)
    except Exception:
        print(
            "\t\t\t--> Local cache not available or incomplete. Downloading from Hugging Face..."
        )
        pair = _from_pretrained(source)
        print("\t\t\t--> Downloaded and cached model.")
        return pair


def compute_embeddings_transformer_encoder(self, texts):
    # Step 1: Chunk the texts respecting token limits
    flat_texts, chunks_dict = flatten_text_chunks(
        texts, self.nlp, tokenizer=self.tokenizer, max_tokens=self.max_tokens
    )

    # Step 2: Tokenize the chunks
    inputs = self.tokenizer(
        flat_texts, return_tensors="pt", truncation=True, padding=True
    )

    # Step 3: Send inputs to correct device
    inputs = {key: value.to(self.device) for key, value in inputs.items()}

    # Step 4: Forward pass
    with torch.no_grad():
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    # Step 5: Bring embeddings back to CPU if needed
    embeddings = embeddings.cpu().tolist() if self.device == "cuda" else embeddings

    # Step 6: Reinsert embeddings into chunk metadata
    return add_embeddings_to_chunks_dict(embeddings, chunks_dict)
