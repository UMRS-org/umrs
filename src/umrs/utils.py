import hashlib
import torch
################################## md5 ##################################

def compute_md5(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()

################################## Chunking texts ##################################

def chunk_text_by_max_tokens(doc, tokenizer, max_tokens):
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
        - num_tokens: number of tokens in the chunk
        - md5: md5 checksum of the chunk
    """
    chunks = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        # Tokenize the sentence and get the mapping between tokens and character offsets
        encoding = tokenizer(sentence_text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        n_tokens = len(input_ids)
        if n_tokens <= max_tokens:
            # If the sentence fits in the limit, use it directly
            start = sent.start_char
            end = sent.end_char
            chunks.append({
                "text": sentence_text,
                "start": start,
                "end": end,
                "num_tokens": n_tokens,
                "md5": compute_md5(sentence_text)
            })
        else:
            # Otherwise, split the sentence into subchunks of max_tokens
            for i in range(0, len(input_ids), max_tokens):
                sub_ids = input_ids[i:i + max_tokens]
                sub_offsets = offsets[i:i + max_tokens]

                # Get the character range covered by the token span
                sub_start = sub_offsets[0][0]
                sub_end = sub_offsets[-1][1]

                # Extract the substring using the token offset range
                sub_text = sentence_text[sub_start:sub_end].strip()
                global_start = sent.start_char + sub_start
                global_end = sent.start_char + sub_end

                if sub_text:
                    chunks.append({
                        "text": sub_text,
                        "start": global_start,
                        "end": global_end,
                        "num_tokens": len(sub_ids),
                        "md5": compute_md5(sub_text)
                    })

    return chunks
def merge_chunks_by_tokens(chunks, max_tokens, tokenizer, original_text):
    """
    Merge consecutive chunks as long as the total token count stays under max_tokens,
    using the original text to preserve inter-chunk separators.

    Args:
        chunks: list of dicts with 'start', 'end', 'text', 'num_tokens'
        max_tokens: maximum number of tokens per merged chunk
        tokenizer: HuggingFace tokenizer
        original_text: the full original input text

    Returns:
        A new list of merged chunks
    """
    merged_chunks = []
    current_chunk = None
    current_token_count = 0

    for chunk in chunks:
        if current_chunk is None:
            current_chunk = chunk.copy()
            current_token_count = chunk["num_tokens"]
        else:
            # Text between current chunk and new chunk
            inter_text = original_text[current_chunk["end"]-1:chunk["start"]]
            inter_tokens = tokenizer(inter_text, add_special_tokens=False)["input_ids"]
            new_total_tokens = current_token_count + len(inter_tokens) + chunk["num_tokens"] + 1

            if new_total_tokens <= max_tokens:
                # Update current_chunk
                current_chunk["end"] = chunk["end"]
                current_chunk["num_tokens"] = new_total_tokens
                current_chunk["text"] = original_text[current_chunk["start"]:current_chunk["end"]]
                current_chunk["md5"] = compute_md5(current_chunk["text"])
                current_token_count = new_total_tokens
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk.copy()
                current_token_count = chunk["num_tokens"]

    if current_chunk:
        merged_chunks.append(current_chunk)

    return merged_chunks

def flatten_text_chunks(texts, nlp, tokenizer, max_tokens):
    """
    Splits a list of texts into smaller chunks that respect the token limit of a transformer model.

    Args:
        texts (list of str): List of raw input texts.
        nlp (spacy.Language): spaCy language model for sentence segmentation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to compute token length.
        max_tokens (int): Maximum number of tokens allowed per chunk.

    Returns:
        tuple:
            - flat_chunks (list of str): All chunks flattened in a single list (for batch embedding).
            - chunks_by_text (list of list of dict): Original text split into chunks, each with metadata.
    """
    flat_chunks = []
    chunks_by_text = []
    for text in texts:
        doc = nlp(text)
        chunks = chunk_text_by_max_tokens(doc, tokenizer, max_tokens)
        chunks = merge_chunks_by_tokens(chunks, max_tokens, tokenizer, text)
        flat_chunks.extend([chunk["text"] for chunk in chunks])
        chunks_by_text.append(chunks)
    return flat_chunks, chunks_by_text

def add_embeddings_to_chunks_dict(embeddings, list_dict):
    """
    Adds computed embeddings back into the corresponding text chunks.

    Args:
        embeddings (list or tensor): List of embedding vectors (must match the number of chunks).
        list_dict (list of list of dict): List of chunk lists (as returned by `flatten_text_chunks`).

    Returns:
        list of list of dict: The same structure as input, but with an "embedding" field added to each chunk.
    """
    index = 0
    for chunk_list in list_dict:
        for chunk in chunk_list:
            chunk["embedding"] = embeddings[index]
            index += 1
    return list_dict


def zero_out_embeddings_for_empty_objects(embeddings, sizes, modality_name):
    """
    Replaces embeddings with zero vectors where the corresponding object size is 0.

    Args:
        embeddings (list of list of dict): Each dict should contain an 'embedding' and 'text' field.
        sizes (list): A flat list of object sizes, one per chunk (same length as total number of chunks).
        modality_name (str): Use only for 'text' or 'text file' modalities.

    Returns:
        None. Operates in-place on `embeddings`.
    """
    if modality_name not in ("text", "text file"):
        return  # Currently handles only textual modalities

    # Get dimension from the first embedding (supports list/np/tensor)
    sample = embeddings[0][0]["embedding"]
    if hasattr(sample, "shape") and getattr(sample, "shape") is not None:
        try:
            embedding_dim = int(sample.shape[0])
        except (AttributeError, IndexError, TypeError):
            embedding_dim = len(sample)  # fallback
    else:
        embedding_dim = len(sample)

    # Use plain Python list to ensure DB serialization robustness
    zero_vec = [0.0] * embedding_dim

    index = 0
    for chunk_list in embeddings:
        for chunk in chunk_list:
            if sizes[index] == 0:
                chunk["embedding"] = zero_vec
                chunk["text"] = ''
                chunk["md5"] = 'd41d8cd98f00b204e9800998ecf8427e' #md5 of empty string
                chunk["start"] = 0
                chunk["end"] = 0
                chunk["num_tokens"] = 0
            index += 1

            
