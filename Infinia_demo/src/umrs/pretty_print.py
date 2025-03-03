import pandas as pd
from tabulate import tabulate
def pretty_print_cfg(cfg_dict):
    # Create an empty DataFrame with predefined columns
    df = pd.DataFrame(columns=["Bucket alias", "Username", "Experience", "Modality", "Model", "Store Embeddings", "Metadata"])
    for key, entry in cfg_dict.items():
        bucket_alias = entry["bucket_alias"]
        username = entry["username"]
        model_name = entry["model"].model_name
        store_embeddings_flag = entry["store_embeddings_flag"]
        modality = entry["modality_name"]
        xp = entry["xp_name"]
        metadata = entry["metadata"]
        metadata_str= ""
        for mkey, mentry in metadata.items():
            name = mentry["name"];
            type = mentry["type"];
            if not metadata_str == "":
                metadata_str += " ; "
            metadata_str += f"{name} ({type})"

        df.loc[len(df)] = [bucket_alias, username, xp, modality, model_name, store_embeddings_flag,  metadata_str]

    print(tabulate(df, headers='keys', tablefmt='psql'))
