
import spacy
import torch 

from typing import Optional

        


class UmrsModel:
    def __init__(self, model_name: str, type: str, return_type: str, embedding_vector_size: int, max_tokens: Optional[int], model, input_type: str, compute_func, modality, spacy_model, tokenizer, device = None):
        self.model_name = model_name
        self.type = type
        self.return_type = return_type
        self.embedding_vector_size = int(embedding_vector_size)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.input_type = input_type
        self.model = model
        self.modality = modality
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = tokenizer
        if self.model and hasattr(self.model, "to"):
            self.model.to(self.device)
        if compute_func:
            self.compute = compute_func.__get__(self)  # Bind the method to the instance
        else:
            self.compute = None
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
    device = None

    spacy_model = model_cfg.get("spacy_model", None)
    model_mode = model_cfg.get("model_mode", None)
    cfg_file = model_cfg.get("model_cfg_file", None)
    checkpoint_path = model_cfg.get("checkpoint_path", None)


    if model_type == "SentenceTransformer":
        from .models.sentence_transformers import load_sentence_transformers_model, sentence_transformers 
        model = load_sentence_transformers_model(model_name)
        type = "loaded"
        tokenizer = model._first_module().tokenizer
        max_tokens = getattr(tokenizer, "model_max_length", None)
        compute_func, input_type = sentence_transformers(modality);
    elif model_type == "Resnet-50":
        from .models.resnet_50 import load_resnet50, compute_embeddings_resnet_50
        model = load_resnet50(checkpoint_path)
        type = "loaded"
        compute_func = compute_embeddings_resnet_50
        input_type = "image"
    elif model_type == "TransformerEncoder":
        from .models.transformer_encoder import load_transformer_encoder, compute_embeddings_transformer_encoder
        model, tokenizer = load_transformer_encoder(model_name)
        type = "loaded"
        max_tokens = tokenizer.model_max_length
        compute_func = compute_embeddings_transformer_encoder
        input_type = "text"
    elif model_type == "HMAE" :
        from .models.masked_autoencoder import load_masked_autoencoder_infer, infer_masked_autoencoder
        model, device = load_masked_autoencoder_infer(cfg_file, checkpoint_path, model_name)
        type = "loaded"
        compute_func = infer_masked_autoencoder(modality, cfg_file)
        input_type = "raw_image"
    elif model_type == "MAE" :
        from .models.masked_autoencoder import load_masked_autoencoder, compute_embeddings_masked_autoencoder
        model = load_masked_autoencoder(cfg_file, checkpoint_path)
        type = "loaded"
        print("MAE model loaded")
        compute_func = compute_embeddings_masked_autoencoder
        input_type = "raw_image"
    elif model_type.startswith("openai://"):
        # Import OpenAI-compatible embedding utilities and vLLM liveness / capability checks
        from .models.openai_api import make_openai_compute_embeddings, is_server_alive, get_max_tokens_from_vllm

        # Import HuggingFace tokenizer to locally mirror the server-side tokenization logic
        from transformers import AutoTokenizer

        # Extract the actual server URL from the openai:// pseudo-scheme
        model_url = model_type[len("openai://"):]

        # Query the vLLM server to discover the maximum supported token length for this model
        max_tokens = get_max_tokens_from_vllm(model_url, model_name)

        # Load the tokenizer corresponding to the embedding model to perform local chunking when necessary
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Fail fast if the OpenAI-compatible (vLLM) server is not reachable
        if not is_server_alive(model_url):
            raise RuntimeError(
                f"Cannot reach OpenAI-compatible server at {model_url}.\n"
                "Check that it is running and reachable from UMRS server."
            )

        # No local model object is instantiated since inference is delegated to the remote server
        model = None
        type = "openai"
        # Build a closure that captures (url, model_name) and exposes a uniform compute_embeddings API
        compute_func = make_openai_compute_embeddings(model_url, model_name)

        # Declare that this embedding model expects raw text as input
        input_type = "text"
    elif model_type.startswith("rivault://"):
        model = model_cfg
        model["model_type"] = model_type[len("rivault://"):]
        type = "rivault"
        input_type=None
        compute_func=None
      
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    # Create and return the EmbeddingModel object
    return UmrsModel(model_name, type, return_type, vector_size, max_tokens, model, input_type, compute_func, modality, spacy_model, tokenizer, device = device)

