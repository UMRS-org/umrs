import pandas as pd
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
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
