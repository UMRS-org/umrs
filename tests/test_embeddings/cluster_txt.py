#!/usr/bin/python3
import ast  # To safely convert strings to lists
import os

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


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
    # select s3_object, embedding from image;
    cur.execute(f"SELECT _content, embedding FROM {table};")
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
            embedding = np.array(
                ast.literal_eval(embedding_str)
            )  # Convert string to list of floats
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

        # Store the closest text to the centroid
        centroid_texts.append(texts[closest_idx])

    return centroid_texts


def query_db_with_embeddings_txt(model, conn, cur, query_text, top_k=5):
    print(f"Searching table '{table}' for {query_text}...")

    query_embedding = model.encode(query_text, convert_to_tensor=False).tolist()

    # Perform similarity search using L2 distance
    cur.execute(
        """
        SELECT _content
        FROM text
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """,
        (query_embedding, top_k),
    )

    # Fetch results
    results = cur.fetchall()

    # Extract IDs and texts
    texts = [row[0] for row in results]

    print(f'For "{query_text}":')
    for text in texts:
        print(f"\t{os.path.basename(text)}")

    return texts


def load_sentence_transformers_model(model_name):

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        model = SentenceTransformer(model_name)
    return model


sql_username = "usr1_2"
db_name = "text_and_images_usr1_2"
table = "text"

DB_HOST = "localhost"
DB_PORT = "5432"
DB_SUPERUSER = "postgres"  # Use a superuser or admin account
DB_NAME = db_name  # Replace with the target database
DB_USER = sql_username  # Replace with the target user
DB_USER_PASSWORD = "ejeannot"

print("Connecting to DB")
# Connect to PostgreSQL default database (`postgres`)
conn = psycopg2.connect(
    dbname=DB_NAME, user=sql_username, host=DB_HOST, port=DB_PORT, password=DB_USER_PASSWORD
)
cur = conn.cursor()

centroid_texts = cluster_from_embeddings(conn, cur, table, nb_clusters=4)
for text in centroid_texts:
    print(f"\t{os.path.basename(text)}")

print("Loading model...")
model = load_sentence_transformers_model("all-mpnet-base-v2")

query_db_with_embeddings_txt(model, conn, cur, "I love eating apple", top_k=3)
query_db_with_embeddings_txt(model, conn, cur, "My computer is a Mac", top_k=3)
query_db_with_embeddings_txt(model, conn, cur, "In the Bible Eve eats the forbiden fruit", top_k=3)
