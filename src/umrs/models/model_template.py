'''
# How to Import a New Model in UMRS

Copy this template:

1. Create a function to load the model.

2. Create a function to compute the embedding based on this model (taking the data/file as a parameter).  
   Since this function will be part of a class, the model will be called using `self.model`.

3. In `model_loading.py`:  
   - Add the model loading using the function you created to load the model.  
   - Attach the function to compute the embedding to the `EmbeddingModel` class. There are two supported ways to compute embeddings:  
     - Using a tensor of data as in `sentence_transformer.py`.  
     - Or a list of files as in `masked_autoencoder.py`.  
   - Be sure that all values of `"model_name"`, `"vector_size"`, `"max_tokens"`, `"model"`, `"input_type"`, `"compute_embeddings_func"`, `"modality"`, `"spacy_model"`, `"tokenizer"`  
     are correctly set before calling the `EmbeddingModel` constructor that is returned.

4. Create or update a user configuration file by setting the values of:  
   - `"model_name"`  
   - `"vector_size"`  
   - `"model_type"`  
   And eventually:  
   - `"checkpoint_path"`  
   - `"model_cfg_file"`  
   Also update other parts of the configuration for metadata and the datatype handled, if necessary.

5. If you have a new modality, add it in `umrs/modality.py`, update `umrs/json_utils.py`, and check how to store it in `umrs/sql.py` (see the function `store_embeddings_in_db`). You should also customize the `client.py` (the `process_file` function).

6. In `server_umrs.py`, check `compute_and_store_embedding()` to see if you are using the file or the tensor method for your modality.

7. Create tests and test both `upload_file` and `upload_dir` commands from the client side. Look at the database to verify the embeddings are correctly stored.

'''



################################ ModelName  ###########################
def load_model_name(model_name):
    pass
    return model, tokenizer


def compute_embeddings_model_name(self, tensor_or_file_list):
    pass
    return embeddings
