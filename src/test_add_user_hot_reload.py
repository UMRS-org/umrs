import sys
import json
from client_umrs_api import UMRSApiClient

# Example user config (minimal, adjust as needed)
user_config = {
    "username": "testuser1",
    "fullname": "Test User 1",
    "s3_configuration_file": "/root/.s3cfg",
    "experiences": [
        {
            "xp_name": "Test Experience",
            "sql_username": "testuser1",
            "modalities": [
                {
                    "modality_name": "text",
                    "s3": {
                        "bucket_alias": "test-bucket-1",
                        "data": {"type": "inline", "mime_type": "text/plain"}
                    },
                    "model": {
                        "model_name": "intfloat/e5-base-v2",
                        "vector_size": "768",
                        "model_type": "TransformerEncoder",
                        "spacy_model": "en_core_web_sm"
                    },
                    "sql": {
                        "store_embeddings": "yes",
                        "table_name": "text",
                        "meta_data": [
                            {"name": "keywords", "type": "TEXT"}
                        ]
                    }
                }
            ]
        }
    ]
}

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://localhost:8443"
    client = UMRSApiClient(base_url)
    try:
        result = client.add_user_hot_reload(user_config)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
