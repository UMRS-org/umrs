import spacy
import torch


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        type: str,
        return_type: str,
        embedding_vector_size: int,
        max_tokens: int | None,
        model,
        input_type: str,
        compute_embeddings,
        modality,
        spacy_model,
        tokenizer,
    ):
        self.model_name = model_name
        self.type = type
        self.return_type = return_type
        self.embedding_vector_size = int(embedding_vector_size)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.input_type = input_type
        self.model = model
        self.modality = modality
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        if self.model and hasattr(self.model, "to"):
            self.model.to(self.device)
        if compute_embeddings:
            self.compute_embeddings = compute_embeddings.__get__(
                self
            )  # Bind the method to the instance
        else:
            self.compute_embeddings = None
        if spacy_model:
            self.nlp = spacy.load(spacy_model, disable=["ner", "tagger", "parser"])
            self.nlp.add_pipe("sentencizer")  # simple sentence seperator
        else:
            self.nlp = None


################################ Models loading  ###########################
#  Load the Model Dynamically Based on Type
def load_model(model_cfg, modality, task=None):
    model_name = model_cfg["model_name"]
    vector_size = model_cfg["vector_size"]
    model_type = model_cfg["model_type"]
    return_type = model_cfg["return_type"]

    tokenizer = None
    max_tokens = None
    spacy_model = model_cfg.get("spacy_model", None)
    cfg_file = model_cfg.get("model_cfg_file", None)
    checkpoint_path = model_cfg.get("checkpoint_path", None)

    if model_type == "SentenceTransformer":
        from .sentence_transformers import (
            load_sentence_transformers_model,
            sentence_transformers,
        )

        model = load_sentence_transformers_model(model_name)
        type = "loaded"
        tokenizer = model._first_module().tokenizer
        max_tokens = getattr(tokenizer, "model_max_length", None)
        compute_embeddings_func, input_type = sentence_transformers(modality)
    elif model_type == "Resnet-50":
        from .resnet_50 import compute_embeddings_resnet_50, load_resnet50

        model = load_resnet50(checkpoint_path)
        type = "loaded"
        compute_embeddings_func = compute_embeddings_resnet_50
        input_type = "image"
    elif model_type == "TransformerEncoder":
        from .transformer_encoder import (
            compute_embeddings_transformer_encoder,
            load_transformer_encoder,
        )

        model, tokenizer = load_transformer_encoder(model_name)
        type = "loaded"
        max_tokens = tokenizer.model_max_length
        compute_embeddings_func = compute_embeddings_transformer_encoder
        input_type = "text"
    elif model_type == "MAE":
        from .masked_autoencoder import (
            compute_embeddings_masked_autoencoder,
            load_masked_autoencoder,
        )

        model = load_masked_autoencoder(cfg_file, checkpoint_path)
        type = "loaded"
        compute_embeddings_func = compute_embeddings_masked_autoencoder
        input_type = "raw_image"
    elif model_type == "HMAE":
        from .hmae_inference import load_hmae_model, make_hmae_compute_func

        model, device, hmae_config = load_hmae_model(cfg_file, checkpoint_path)
        type = "loaded"
        compute_embeddings_func = make_hmae_compute_func(modality, cfg_file, task=task)
        input_type = "raw_image"
        # Store config on the EmbeddingModel for use by the compute function
        _hmae_config = hmae_config
    elif model_type == "SAM3":
        from .sam3_segmentation import compute_sam3_segmentation, load_sam3_model

        sam3_model, sam3_processor, sam3_device = load_sam3_model(model_name)
        model = sam3_model
        type = "loaded"
        compute_embeddings_func = compute_sam3_segmentation
        input_type = "image"
        _sam3_processor = sam3_processor
    elif model_type.startswith("openai://"):
        # Import OpenAI-compatible embedding utilities and vLLM liveness / capability checks
        # Import HuggingFace tokenizer to locally mirror the server-side tokenization logic
        from transformers import AutoTokenizer

        from .openai_api import (
            get_max_tokens_from_vllm,
            is_server_alive,
            make_openai_compute_embeddings,
        )

        # Extract the actual server URL from the openai:// pseudo-scheme
        model_url = model_type[len("openai://") :]

        # Extract optional API key (set by rivault launch flow for LiteLLM auth)
        api_key = model_cfg.get("api_key")

        # Load the tokenizer first so we have its max_length as fallback
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Query the vLLM server to discover the maximum supported token length for this model.
        # LiteLLM proxy responses lack vLLM-specific fields, so fall back to
        # the tokenizer's model_max_length when no field is exposed.
        max_tokens = get_max_tokens_from_vllm(model_url, model_name, api_key=api_key)
        if max_tokens is None:
            max_tokens = getattr(tokenizer, "model_max_length", 512)
            print(
                f"[INFO] vLLM max_token field not exposed (LiteLLM proxy?); using tokenizer max_length={max_tokens}"
            )

        # Fail fast if the OpenAI-compatible (vLLM) server is not reachable
        if not is_server_alive(model_url, api_key=api_key):
            raise RuntimeError(
                f"Cannot reach OpenAI-compatible server at {model_url}.\n"
                "Check that it is running and reachable from UMRS server."
            )

        # No local model object is instantiated since inference is delegated to the remote server
        model = None
        type = "openai"
        # Build a closure that captures (url, model_name) and exposes a uniform compute_embeddings API
        compute_embeddings_func = make_openai_compute_embeddings(
            model_url, model_name, api_key=api_key
        )

        # Declare that this embedding model expects raw text as input
        input_type = "text"
    elif model_type.startswith("user_script://"):
        from umrs.pipeline_sdk.adapter import EmbeddingAdapter
        from umrs.pipeline_sdk.workflow import WorkflowPipeline

        source_uri = model_type[len("user_script://") :]
        pipeline = WorkflowPipeline(source_uri)
        adapter = EmbeddingAdapter(pipeline)

        model = None
        type = "user_script"

        # Wrap adapter call so EmbeddingModel.__get__ rebinding does not
        # replace the adapter's ``self`` with the EmbeddingModel instance.
        def _user_script_compute(
            _self, s3, s3_keys, path_to_bucket, output_bucket, **kwargs
        ):
            return adapter.compute_embeddings(
                s3, s3_keys, path_to_bucket, output_bucket, **kwargs
            )

        compute_embeddings_func = _user_script_compute
        input_type = "raw_image"  # steps handle their own I/O from S3
        return_type = pipeline.workflow.output.return_type
        vector_size = pipeline.workflow.output.embedding_vector_size or vector_size

    elif model_type.startswith("user_workflow://"):
        from umrs.inference.plugins.user_workflow import (
            UserWorkflow,
            UserWorkflowRunner,
        )
        from umrs.pipeline_sdk.adapter import EmbeddingAdapter
        from umrs.pipeline_sdk.workflow import WorkflowPipeline

        source_uri = model_type[len("user_workflow://") :]
        pipeline = WorkflowPipeline(source_uri)
        uw = UserWorkflow.from_workflow_definition(pipeline.workflow)
        runner = UserWorkflowRunner(uw)

        model = None
        type = "user_workflow"

        # Same __get__ rebinding guard as user_script://
        def _user_workflow_compute(_self, *args, **kwargs):
            return runner.run(*args, **kwargs)

        compute_embeddings_func = _user_workflow_compute
        input_type = "raw_image"
        return_type = pipeline.workflow.output.return_type
        vector_size = pipeline.workflow.output.embedding_vector_size or vector_size

    elif model_type.startswith("pipeline://"):
        model = model_cfg
        model["model_type"] = model_type
        model["_modality"] = modality
        model["_task"] = task
        type = "pipeline"
        input_type = None
        compute_embeddings_func = None

    elif model_type.startswith("rivault://"):
        model = model_cfg
        model["model_type"] = model_type[len("rivault://") :]
        model["_modality"] = modality
        model["_task"] = task
        type = "rivault"
        input_type = None
        compute_embeddings_func = None

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create and return the EmbeddingModel object
    em = EmbeddingModel(
        model_name,
        type,
        return_type,
        vector_size,
        max_tokens,
        model,
        input_type,
        compute_embeddings_func,
        modality,
        spacy_model,
        tokenizer,
    )

    # Attach extra attributes for vision models
    if model_type == "HMAE":
        em._hmae_config = _hmae_config
        em.device = device
    elif model_type == "SAM3":
        em._sam3_processor = _sam3_processor

    return em
