import torch
from umrs.utils import flatten_text_chunks, add_embeddings_to_chunks_dict
from sentence_transformers import SentenceTransformer

################################ Sentence Transformers ###########################
#  Sentence Transformer Loader
_model_cache = {}

def load_sentence_transformers_model(model_name):
    if model_name in _model_cache:
        print(f"\t\t\t--> Model '{model_name}' already loaded...")
        return _model_cache[model_name]

    print(f"\t\t\t--> Loading model '{model_name}'...")
    try :
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        print("\t\t\t--> Local cache not available or incomplete. Downloading from Hugging Face...")
        model = SentenceTransformer(model_name)
        print("\t\t\t--> Downloaded and cached model.")
    
    _model_cache[model_name] = model
    return model;


#  Compute Embeddings for Text
def compute_embeddings_text(self, texts):
    flat_texts, chunks_dict = flatten_text_chunks(texts, self.nlp, tokenizer=self.tokenizer, max_tokens=self.max_tokens)
    embeddings = self.model.encode(flat_texts, convert_to_tensor=(self.device == "cuda"))
    embeddings = embeddings.cpu().tolist() if self.device == "cuda" else embeddings
    return add_embeddings_to_chunks_dict(embeddings, chunks_dict)

#  Compute Embeddings for Images
def compute_embeddings_img(self, images):
    # Ensure images are in RGB format
    images = [img.convert("RGB") for img in images]

    # Compute embeddings
    with torch.no_grad():
        embeddings = self.model.encode(images, convert_to_tensor=(self.device == "cuda"))

    return embeddings.cpu().tolist() if self.device == "cuda" else embeddings


def sentence_transformers(modality):
    if modality == "text":
        compute_embeddings_func = compute_embeddings_text
        input_type = "text"
    elif modality == "image":
        compute_embeddings_func = compute_embeddings_img
        input_type = "image"
    else:
        raise ValueError(f"Unsupported modality '{modality}' for SentenceTransformer model")
    return compute_embeddings_func, input_type
