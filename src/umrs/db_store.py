
def store_db_model(db_model_store, path_to_bucket, bucket_alias, s3_username, db_pool, db_name, table_name, sql_username, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name):
    """Stores data base and model for a given (user, bucket_alias)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry
    
    db_model_store[path_to_bucket] = {"bucket_alias": bucket_alias,
                                      "s3_username": s3_username,
                                      "sql_username": sql_username,
                                      "data_base": db_pool,
                                      "db_name": db_name,
                                      "table_name": table_name,
                                      "model": model,
                                      "store_embeddings_flag": store_embeddings_flag,
                                      "modality_name": modality_name,
                                      "s3_connection": s3_connection,
                                      "metadata": metadata,
                                      "xp_name": xp_name,
                                                  }


def query_db_model(db_model_store, path_to_bucket):
    """Retrieves database and model for a given (user, bucket_alias)."""
    entry = db_model_store.get(path_to_bucket, None)
    if not entry:
        return None, None, None, None, None, None, None
    else:
        return entry["s3_username"], entry["data_base"], entry["table_name"], entry["model"], entry["store_embeddings_flag"], entry["modality_name"], entry["s3_connection"]
