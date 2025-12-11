
import spacy
import torch 

from typing import Optional

        


class EmbeddingModel:
    def __init__(self, model_name: str, return_type: str, embedding_vector_size: int, max_tokens: Optional[int], model, input_type: str, compute_embeddings, modality, spacy_model, tokenizer):
        self.model_name = model_name
        self.return_type = return_type
        self.embedding_vector_size = int(embedding_vector_size)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.input_type = input_type
        self.model = model
        self.modality = modality
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.compute_embeddings = compute_embeddings.__get__(self)  # Bind the method to the instance
        if spacy_model:
            self.nlp = spacy.load(spacy_model, disable=["ner", "tagger", "parser"])
            self.nlp.add_pipe("sentencizer")  # simple sentence seperator
        else:
            self.nlp = None





################################ Models loading  ###########################
#  Load the Model Dynamically Based on Type
def load_model(model_cfg, modality):
    model_name = model_cfg["model_name"]
    vector_size = model_cfg["vector_size"]
    model_type = model_cfg["model_type"]
    return_type  = model_cfg["return_type"]
    
    tokenizer = None
    max_tokens = None
    spacy_model = model_cfg.get("spacy_model", None)
    cfg_file = model_cfg.get("model_cfg_file", None)
    checkpoint_path = model_cfg.get("checkpoint_path", None)


    if model_type == "SentenceTransformer":
        from .models.sentence_transformers import load_sentence_transformers_model, sentence_transformers 
        model = load_sentence_transformers_model(model_name)
        tokenizer = model._first_module().tokenizer
        max_tokens = getattr(tokenizer, "model_max_length", None)
        compute_embeddings_func, input_type = sentence_transformers(modality);
    elif model_type == "Resnet-50":
        from .models.resnet_50 import load_resnet50, compute_embeddings_resnet_50
        model = load_resnet50(checkpoint_path)
        compute_embeddings_func = compute_embeddings_resnet_50
        input_type = "image"
    elif model_type == "TransformerEncoder":
        from .models.transformer_encoder import load_transformer_encoder, compute_embeddings_transformer_encoder
        model, tokenizer = load_transformer_encoder(model_name)
        max_tokens = tokenizer.model_max_length
        compute_embeddings_func = compute_embeddings_transformer_encoder
        input_type = "text"
    elif model_type == "MAE" :
        from .models.masked_autoencoder import load_masked_autoencoder, compute_embeddings_masked_autoencoder
        model = load_masked_autoencoder(cfg_file, checkpoint_path)
        compute_embeddings_func = compute_embeddings_masked_autoencoder
        input_type = "raw_image"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    # Create and return the EmbeddingModel object
    return EmbeddingModel(model_name, return_type, vector_size, max_tokens, model, input_type, compute_embeddings_func, modality, spacy_model, tokenizer)

