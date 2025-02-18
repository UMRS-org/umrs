#!/usr/bin/python3
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from umrs.embeddings import *
import time
from collections import defaultdict
import signal
import torch



def store_db_model(db_model_store, bucket_name, db_pool, table_name, model, modality_name, s3_connection):
    """Stores db_pool and model for a given (user, bucket_name)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry
    
    db_model_store[bucket_name] = {"db_pool": db_pool, "table_name": table_name,  "model": model, "modality_name": modality_name, "s3_connection": s3_connection}


def query_db_model(db_model_store, bucket_name):
    """Retrieves db_pool and model for a given (user, bucket_name)."""
    entry = db_model_store.get(bucket_name, None)
    if not entry:
        return None, None, None
    else:
        return entry["db_pool"], entry["table_name"], entry["model"], entry["modality_name"], entry["s3_connection"]

    
def configure_user(user_cfg):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    db_model_store = defaultdict(dict)
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\tExperience: {xp_name}")
        sql_username  = xp["sql_username"]
        db_pool = create_db_and_get_pool(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            print(f"\t\tModality: {modality_name}")
            bucket_name, s3_connection = configure_s3(modality["s3"], user_s3_configuration_file, xp_name, username)
            table_name, model_name = configure_sql(db_pool, modality["sql"], modality_name)
            model = load_embedding_model(model_name)
            store_db_model(db_model_store, bucket_name, db_pool, table_name, model, modality_name, s3_connection)

    return db_model_store

def compute_image_embeddings(images, model, device="cpu"):
    """
    Compute image embeddings in batch.
    """
    # Convert images to tensors if needed
    images = [img.convert("RGB") for img in images]  # Ensure RGB format
    with torch.no_grad():
        embeddings = model.encode(images, convert_to_tensor=(device == "cuda"))

    return embeddings.cpu().tolist() if device == "cuda" else embeddings



def compute_and_store_embedding(bucket_name, s3_keys, db_pool, table_name, model, device, modality_name, s3_connection):
    """
    Loads data from S3, computes embeddings **in batch**, and stores them in the database.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_keys (list): List of S3 object keys.
        db_pool: Database connection pool.
        table_name (str): Target table for storing embeddings.
        model (SentenceTransformer): Model used for embedding computation.
        device (str): "cpu" or "cuda".
        modality_name (str): "text" or "image".
        s3_connection: Boto3 S3 client.
    """
    objects = load_data_from_s3(s3_connection, bucket_name, s3_keys, modality_name)
    
    if modality_name == "text":
        texts = objects  # List of text documents
        embeddings = model.encode(texts, convert_to_tensor=False).tolist()  # Batch text embeddings
    
    elif modality_name == "image":
        texts = None  # No text for images
        embeddings = compute_image_embeddings(objects, model, device)  # Batch image embeddings
    
    else:
        raise ValueError(f"Unsupported modality: {modality_name}")

    # Store **all** embeddings in batch
    store_embeddings_in_db(db_pool, embeddings, table_name, bucket_name, s3_keys, texts)


def perf_test(s3, bucket_name, modality_name, model, db_pool, table_name, stage_max = 1, n=1):
    response = s3.list_objects_v2(Bucket=bucket_name)

    stage = 1
    keys = []
    total_size = 0
    if "Contents" in response:
        for obj in response["Contents"]:
            object_key = obj["Key"]
            keys.append(object_key)
            res = s3.head_object(Bucket=bucket_name, Key=object_key)
            size = res["ContentLength"]
            # print(f"{size} {object_key}")
            total_size += size
    # print(f"Total size: {total_size}")

    nb_objects = len (keys)
    if stage>stage_max:
        return
    start = 1000*time.perf_counter()
    for _ in range(n):
        for s3_key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
            image = BytesIO(obj["Body"].read())
            # image = Image.open(BytesIO(obj["Body"].read()))
            # load_data_from_s3(s3, bucket_name, s3_key, modality_name)

    end = 1000*time.perf_counter()
    total_time = end - start # ms
    throughput_MB_p_s = (n * total_size /(1024*1024))/(total_time/1000)
    print(f"stage: BytesIO ; nb_objects: {nb_objects*n} ; Total_size: {total_size*n/(1024*1024):.3f}MB ; total_time: {total_time:.3f} ms ; Throughput: {throughput_MB_p_s:.3f} MB/s")
    stage += 1
    if stage>stage_max:
        return
    start = 1000*time.perf_counter()
    for _ in range(n):
        for s3_key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
            # image = BytesIO(obj["Body"].read())
            image = Image.open(BytesIO(obj["Body"].read()))
            # load_data_from_s3(s3, bucket_name, s3_key, modality_name)
    end = 1000*time.perf_counter()

    total_time = end - start # ms
    throughput_MB_p_s = (n * total_size /(1024*1024))/(total_time/1000)
    print(f"stage: BytesIO+PIL ; nb_objects: {nb_objects*n} ; Total_size: {total_size*n/(1024*1024):.3f}MB ; total_time: {total_time:.3f} ms ; Throughput: {throughput_MB_p_s:.3f} MB/s")
    
    stage += 1
    if stage>stage_max:
        return
    start = 1000*time.perf_counter()
    for _ in range(n):
        for s3_key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
            # image = BytesIO(obj["Body"].read())
            image = Image.open(BytesIO(obj["Body"].read()))
            embedding = model.encode(image, convert_to_tensor=False).tolist()
            # load_data_from_s3(s3, bucket_name, s3_key, modality_name)
    end = 1000*time.perf_counter()

    total_time = end - start # ms
    throughput_MB_p_s = (n * total_size /(1024*1024))/(total_time/1000)
    print(f"stage: BytesIO+PIL+Embedding ; nb_objects: {nb_objects*n} ; Total_size: {total_size*n/(1024*1024):.3f}MB ; total_time: {total_time:.3f} ms ; Throughput: {throughput_MB_p_s:.3f} MB/s")
    
    stage += 1
    if stage>stage_max:
        return
    start = 1000*time.perf_counter()
    for _ in range(n):
        for s3_key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
            # image = BytesIO(obj["Body"].read())
            image = Image.open(BytesIO(obj["Body"].read()))
            embedding = model.encode(image, convert_to_tensor=False).tolist()
            store_embeddings_in_db(db_pool, embedding, table_name, bucket_name, s3_key, None, silent=True)
            # load_data_from_s3(s3, bucket_name, s3_key, modality_name)
    end = 1000*time.perf_counter()

    total_time = end - start # ms
    throughput_MB_p_s = (n * total_size /(1024*1024))/(total_time/1000)
    print(f"stage: BytesIO+PIL+Embedding+SQL_Store ; nb_objects: {nb_objects*n} ; Total_size: {total_size*n/(1024*1024):.3f}MB ; total_time: {total_time:.3f} ms ; Throughput: {throughput_MB_p_s:.3f} MB/s")
    
def run_perf_test():
    bucket_name = "ejeannot-text-and-images-ej-bucket-1"
    db_pool, table_name, model, modality_name, s3_connection = query_db_model(db_model_store, bucket_name)
    bucket_name = "tarballs"
    perf_test(s3_connection, bucket_name, "image", model,db_pool, table_name, stage_max = 1)
    bucket_name = "ejeannot-text-and-images-ej-bucket-1"
    perf_test(s3_connection, bucket_name, "image", model, db_pool, table_name, stage_max=1, n=100)
    perf_test(s3_connection, bucket_name, "image", model, db_pool, table_name, stage_max=2, n=100)
    perf_test(s3_connection, bucket_name, "image", model, db_pool, table_name, stage_max=3, n=10)
    perf_test(s3_connection, bucket_name, "image", model, db_pool, table_name, stage_max=4, n=10)



    
if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <JSON config file>")

configure_file = sys.argv[1]
    
# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f"Configuration file \"{configure_file}\" not found.")

# Load JSON content from the file
with open(configure_file, "r", encoding="utf-8") as file:
    content = json.load(file)

# Print loaded content (optional)
print("Configuration loaded successfully.")

# Pretty-print the JSON content
# print(json.dumps(content, indent=4, sort_keys=True, ensure_ascii=False))



for user in content["users"]:
    db_model_store = configure_user(user)




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-->  Models will be run on device: {device}")


print(f"\n######################\nServer configuration done!\n######################\n")

# run_perf_test()

try:
    # loop for new new object to be created
    while True:
        print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")

        s3_objects = wait_for_s3_events()

        keys_per_bucket = defaultdict(list)  # Create a dictionary with lists
        print(f"\t-->Detected {len(s3_objects)} new file(s):")
        

        for s3_object in s3_objects:
            print(f"\t\t-->{s3_object}")
            bucket_name, s3_key = s3_object.replace("s3://", "").split("/", 1)
    
            # Append s3_key to the list associated with the bucket
            if s3_key not in keys_per_bucket[bucket_name]:  # Ensure no duplicates
                keys_per_bucket[bucket_name].append(s3_key)


        for bucket_name in keys_per_bucket.keys():
            db_pool, table_name, model, modality_name, s3_connection = query_db_model(db_model_store, bucket_name)
            s3_keys = keys_per_bucket[bucket_name]
            if model:
                print(f"\t--> Storing to data base in {table_name}")
                compute_and_store_embedding(bucket_name, s3_keys, db_pool, table_name, model, device, modality_name, s3_connection) 
            else:
                print(f"No model associated with this bucket")
        
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully...")
    sys.exit(0)  # ✅ Ensures a clean exit
