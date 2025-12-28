
def store_db_model(db_model_store, path_to_bucket, output_path_to_bucket, bucket_alias, 
                   s3_username, data_base, db_name, table_name, sql_username, model,
                   store_embeddings_flag, modality_name, s3_connection, metadata, xp_name,
                   num_objects, reentrant_lock):
    """Stores data base and model for a given (user, bucket_alias)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry
    
    db_model_store[path_to_bucket] = {
                                        "bucket_alias": bucket_alias,
                                        "s3_username": s3_username,
                                        "sql_username": sql_username,
                                        "data_base": data_base,
                                        "db_name": db_name,
                                        "table_name": table_name,
                                        "model": model,
                                        "store_embeddings_flag": store_embeddings_flag,
                                        "modality_name": modality_name,
                                        "s3_connection": s3_connection,
                                        "metadata": metadata,
                                        "xp_name": xp_name,
                                        "output_path_to_bucket": output_path_to_bucket,
                                        "num_objects": num_objects,
                                        "reentrant_lock": reentrant_lock,
                                        }


def query_db_model(db_model_store, path_to_bucket):
    """Retrieves database and model for a given (user, bucket_alias)."""
    entry = db_model_store.get(path_to_bucket, None)
    if not entry:
        return None, None, None, None, None, None, None, None, None, None
    else:
        return entry["s3_username"], entry["data_base"], entry["table_name"], entry["model"], entry["store_embeddings_flag"], entry["modality_name"], entry["s3_connection"], entry['db_name'], entry["output_path_to_bucket"], entry["reentrant_lock"]

def find_entry_by_output_bucket(db_model_store, output_path_to_bucket):
    for path_to_bucket, entry in db_model_store.items():
        if entry.get("output_path_to_bucket") == output_path_to_bucket:
            return path_to_bucket, entry
    return None, None