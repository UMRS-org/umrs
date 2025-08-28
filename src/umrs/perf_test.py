#!/usr/bin/python3

import time  # For performance timing using time.perf_counter
from io import BytesIO  # To read binary data from S3 objects
from PIL import Image  # For image loading and manipulation
import boto3  # AWS SDK for Python to interact with S3

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



    
