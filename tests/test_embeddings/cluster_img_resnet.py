#!/usr/bin/python3 
import psycopg2
from psycopg2 import sql, pool
import numpy as np
from sklearn.cluster import KMeans
import ast  # To safely convert strings to lists
from PIL import Image
import os, sys
# Add the parent directory of 'umrs' to sys.path
sys.path.append(os.path.abspath('.'))
from umrs.inference import *
from types import SimpleNamespace

def cluster_from_embeddings(conn, cur, table, nb_clusters=3):
    """
    Performs K-Means clustering on text embeddings retrieved from PostgreSQL 
    and returns one representative text per cluster.

    Args:
        conn: Active PostgreSQL connection.
        cur: Cursor for executing queries.
        nb_clusters (int): Number of clusters.

    Returns:
        list: One representative text from each cluster.
    """

    print(f"Clustering data from table '{table}' using kmeans...")
    
    # Fetch all embeddings and their corresponding texts from the database
    #select s3_object, embedding from image;
    cur.execute(f"SELECT s3_object, embedding FROM {table};")
    results = cur.fetchall()

    if not results:
        print("No embeddings found in the database.")
        return []

    # Extract texts and process embeddings
    texts = [row[0] for row in results]  # Extract text values
    embeddings = []

    for row in results:
        embedding_str = row[1]  # Extract the stored vector (string format)
        if isinstance(embedding_str, str):
            embedding = np.array(ast.literal_eval(embedding_str))  # Convert string to list of floats
        else:
            embedding = np.array(embedding_str)  # Already a list
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings)  # Convert list of arrays to a 2D NumPy array

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=nb_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Find one representative text per cluster (closest to centroid)
    centroid_texts = []
    for cluster_id in range(nb_clusters):
        # Get indices of points in the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            continue  # Skip empty clusters

        # Find the point closest to the cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        closest_idx = min(cluster_indices, key=lambda i: np.linalg.norm(embeddings[i] - centroid))
        centroid_texts.append(texts[closest_idx])

    return centroid_texts

def query_db_with_embeddings_image(model, conn, cur, image_file, top_k=5):
    print(f"Searching table '{table}' for {image_file}...")

    images = []
    images.append( Image.open(image_file))

    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    obj = SimpleNamespace(model=model, device=device)

    query_embedding = compute_embeddings_resnet_50(obj, images).squeeze().tolist()

    # Perform similarity search using L2 distance
    cur.execute("""
        SELECT s3_object
        FROM image
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, top_k))

    # Fetch results
    results = cur.fetchall()
    
    # Extract IDs and texts
    texts = [row[0] for row in results]

    print(f"For {image_file}:")
    for text in texts:
        print(f"\t{os.path.basename(text)}")
    
    return texts



sql_username = "ejeannot"
db_name =  "images_and_texts_ejeannot"
table = "image"

DB_HOST = "localhost"
DB_PORT = "5432"
DB_SUPERUSER = "ejeannot"  # Use a superuser or admin account
DB_NAME = db_name  # Replace with the target database
DB_USER = sql_username  # Replace with the target user

print("Connecting to DB")
# Connect to PostgreSQL default database (`postgres`)
conn = psycopg2.connect(dbname=DB_NAME, user=DB_SUPERUSER, host=DB_HOST, port=DB_PORT)
cur = conn.cursor()

centroid_texts = cluster_from_embeddings(conn, cur, table, nb_clusters = 3)
for text in centroid_texts:
        print(f"\t{os.path.basename(text)}")

print("Loading model...")
model = load_resnet50("../../data/models/nvidia_resnet50_200821.pth.tar")
    
query_db_with_embeddings_image(model, conn, cur, "../../data/img_test/cat.jpeg", top_k=12)
query_db_with_embeddings_image(model, conn, cur, "../../data/img_test/dog.jpeg", top_k=11)
query_db_with_embeddings_image(model, conn, cur, "../../data/img_test/eagle.jpeg", top_k=11)

