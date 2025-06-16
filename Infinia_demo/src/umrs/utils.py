import hashlib

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
        - length: character length of the chunk
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

        if len(input_ids) <= max_tokens:
            # If the sentence fits in the limit, use it directly
            start = sent.start_char
            end = sent.end_char
            chunks.append({
                "text": sentence_text,
                "start": start,
                "end": end,
                "md5": compute_md5(sentence_text)
            })
        else:
            # Otherwise, split the sentence into subchunks of max_tokens
            print("Splitings...")
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
                        "length": global_end - global_start,
                        "md5": compute_md5(sub_text)
                    })

    return chunks

def flatten_text_chunks(texts, nlp, tokenizer, max_tokens):
    flat_chunks = []
    chunks_by_text = []
    for text in texts:
        doc = nlp(text)
        chunks = chunk_text_by_max_tokens(doc, tokenizer, max_tokens)
        flat_chunks.extend([chunk["text"] for chunk in chunks])
        chunks_by_text.append(chunks)
    return flat_chunks, chunks_by_text

def add_embeddings_to_chunks_dict(embeddings, list_dict):
    index = 0
    for chunk_list in list_dict:
        for chunk in chunk_list:
            chunk["embedding"] = embeddings[index]
            index += 1
    return list_dict
