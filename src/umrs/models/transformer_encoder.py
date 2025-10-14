from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging
from umrs.utils import flatten_text_chunks, add_embeddings_to_chunks_dict
import torch

################################ TransformerEncoder  ###########################
def load_transformer_encoder(model_name):
   

    logging.set_verbosity_error()  # avoid verbose warning
    print(f"\t\t\t--> Loading model '{model_name}'...")

    try:
        # 1. Try to load locally
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        print("\t\t\t--> Local cache not available or incomplete. Downloading from Hugging Face...")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("\t\t\t--> Downloaded and cached model.")

    return model, tokenizer


def compute_embeddings_transformer_encoder(self, texts):
     # Step 1: Chunk the texts respecting token limits
    flat_texts, chunks_dict = flatten_text_chunks(
        texts,
        self.nlp,
        tokenizer=self.tokenizer,
        max_tokens=self.max_tokens
    )

    # Step 2: Tokenize the chunks
    inputs = self.tokenizer(flat_texts, return_tensors="pt", truncation=True, padding=True)

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
