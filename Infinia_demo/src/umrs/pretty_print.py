import pandas as pd
import sys
from tabulate import tabulate
def pretty_print_cfg(cfg_dict):
    # Create an empty DataFrame with predefined columns
    df = pd.DataFrame(columns=["Bucket alias", "S3 user", "S3 path", "Experience", "Modality", "Database", "SQL table", "SQL user", "Model", "Embeds?", "Metadata"])
    for key, entry in cfg_dict.items():
        bucket_alias = entry["bucket_alias"]
        s3_username = entry["s3_username"]
        sql_username = entry["sql_username"]
        model_name = entry["model"].model_name
        store_embeddings_flag = entry["store_embeddings_flag"]
        modality = entry["modality_name"]
        table_name = entry["table_name"]
        xp = entry["xp_name"]
        db_name = entry["db_name"]
        metadata = entry["metadata"]
        metadata_str= ""
        for mkey, mentry in metadata.items():
            name = mentry["name"];
            type = mentry["type"];
            if not metadata_str == "":
                metadata_str += " ; "
            metadata_str += f"{name} ({type})"

        df.loc[len(df)] = [bucket_alias, s3_username, key, xp, modality, db_name, table_name, sql_username, model_name, store_embeddings_flag,  metadata_str]

    df.sort_values(by=["S3 user", "Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("\nServer confuguration:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))


def pretty_print_user_cfg(cfg_dict):
    
    df = pd.DataFrame(columns=["Modality", "Bucket alias", "Experience", "Bucket path", "DB name", "SQL user"])
    username = None
    for key, entry in cfg_dict.items():
        bucket_alias = key
        bucket_path = entry['path_to_bucket']
        xp_name = entry['xp_name']
        modality = entry['modality_name']
        sql_username = entry['sql_username']
        db_name = entry['db_name']

        user = entry['username']
        if not username:
            username = user
        elif user != username:
            sys.exit(f"user mismatch between {username} and {user}")
        
        df.loc[len(df)] = [modality, bucket_alias, xp_name, bucket_path, db_name, sql_username]
    df.sort_values(by=["Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)
        
    print(f"Configuration for S3 user '{username}':")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

def pretty_print_recompute(recompute_dict):
    df = pd.DataFrame(columns=["Bucket alias", "S3 user",  "Bucket path", "Modality", "Model", "Objects", "Size (MB)", "Duration (s)"])

    for path_to_bucket, entry in recompute_dict.items():
        bucket_alias = entry["bucket_alias"]
        modality = entry["modality_name"]
        duration = entry["duration"]
        num_objects = entry["nb_objects"]
        s3_user = entry["s3_username"]
        model=  entry["model"].model_name
        size = entry["total_size_MB"]

        df.loc[len(df)] = [bucket_alias, s3_user,  path_to_bucket, modality, model, num_objects, size, duration]

    df.sort_values(by=["S3 user", "Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\nRecompute Summary:")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

    import pandas as pd
from tabulate import tabulate

def pretty_print_check(check_dict):
    df = pd.DataFrame(columns=["Bucket alias", "S3 user", "Bucket path", "Modality", "Model",
                               "Objects", "Size (MB)", "Matches", "Mismatches", "Not Found", "Errors"])

    for path_to_bucket, entry in check_dict.items():
        bucket_alias   = entry["bucket_alias"]
        s3_user        = entry["s3_username"]
        modality       = entry["modality_name"]
        model_name     = entry["model"].model_name if hasattr(entry["model"], "model_name") else str(entry["model"])
        num_objects    = entry["nb_objects"]
        size_mb        = entry["total_size_MB"]
        nb_match       = entry["matches"]
        nb_mismatch    = entry["Mismatches"]
        nb_not_found   = entry["Not_found"]
        nb_error       = entry["Errors"]

        df.loc[len(df)] = [
            bucket_alias,
            s3_user,
            path_to_bucket,
            modality,
            model_name,
            num_objects,
            size_mb,
            nb_match,
            nb_mismatch,
            nb_not_found,
            nb_error
        ]

    df.sort_values(by=["S3 user", "Bucket alias"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\nObject Check Summary:")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
