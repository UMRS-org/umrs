#!/usr/bin/python3
"""Batch embedding compute pipeline (R6.1 ownership boundary, 2026-08-23).

This module is the ONLY owner of batch/S3 object embedding compute:
model-driven (EmbeddingModel interface), S3 data I/O, DB persistence.
It is NOT the query-time retrieval embedder — live on the frontend, see
``umrs.api.rag_service.PrefixedSentenceTransformerEmbeddings`` (sync ST
client, different process and latency class). Not to be merged (R6.1).
"""

# import psycopg2
import json
import os

import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import ast  # To safely convert strings to lists
import os
import time as _time

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from umrs.metrics import METRICS
from umrs.shared.storage.s3 import load_data_from_s3
from umrs.shared.utils import zero_out_embeddings_for_empty_objects

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Modalities where the model handles its own S3 I/O (data loading + output
# writing) rather than receiving pre-loaded tensors.
# DEPRECATED: kept for backward compatibility; prefer task-based routing via
# DIRECT_COMPUTE_TASKS.
DIRECT_COMPUTE_MODALITIES = frozenset(
    {
        "tomography",
        "tomography_embedding",
        "tomography_reconstruction",
        "tomography_mae_reconstruction",
        "tomography_features_extraction",
        "segmentation",
    }
)

# Tasks that require direct-compute (model handles its own S3 I/O).
DIRECT_COMPUTE_TASKS = frozenset(
    {
        "segmentation",
        "reconstruction",
        "mae_reconstruction",
        "feature_extraction",
        "detection",
        "classification",
    }
)

# Model types that always do their own I/O regardless of modality name.
DIRECT_COMPUTE_MODEL_TYPES = frozenset({"user_script", "bentoml"})


def compute_embeddings_core(
    model,
    modality_name,
    s3_connection,
    path_to_bucket,
    s3_keys,
    output_bucket=None,
    task=None,
    skip_existing=True,
):
    """Core embedding computation pipeline.

    For *direct-compute* tasks the model handles its own data loading and
    output writing.  For standard tasks (embedding) data is loaded from S3
    and passed to the model.

    Args:
        model: ``EmbeddingModel`` instance (must expose ``compute_embeddings``).
        modality_name: Pure modality string (e.g. ``"text"``, ``"image"``,
            ``"volume"``).
        s3_connection: boto3-compatible S3 client.
        path_to_bucket: Source bucket name.
        s3_keys: List of S3 object keys to process.
        output_bucket: Destination bucket for models that produce output
            files (``None`` when not applicable).
        task: Processing task (e.g. ``"embedding"``, ``"segmentation"``).
            When ``None``, falls back to legacy modality-name routing.
        skip_existing: If True, skip computation when output already exists
            in output_bucket. Only applies to direct-compute tasks.

    Returns:
        ``(embeddings, checksums, metadata, saved_s3_keys)`` – four parallel
        lists.  ``saved_s3_keys`` is a list of ``None`` for modalities that
        don't produce output files.
    """
    model_type = getattr(model, "type", None)

    # Determine if this is a direct-compute path.
    is_direct = (
        model_type in DIRECT_COMPUTE_MODEL_TYPES
        or (task is not None and task in DIRECT_COMPUTE_TASKS)
        or (task is None and modality_name in DIRECT_COMPUTE_MODALITIES)
    )

    if is_direct:
        embeddings, checksums, metadata, saved_s3_keys = model.compute_embeddings(
            s3_connection,
            s3_keys,
            path_to_bucket,
            output_bucket,
            skip_existing=skip_existing,
        )
        return embeddings, checksums, metadata, saved_s3_keys

    # Standard path: load data first, then compute embeddings.
    saved_s3_keys = [None] * len(s3_keys)

    objects, metadata, checksums, sizes, has_zero = load_data_from_s3(
        s3_connection,
        path_to_bucket,
        s3_keys,
        modality_name,
    )

    embeddings = model.compute_embeddings(objects)
    if has_zero:
        zero_out_embeddings_for_empty_objects(embeddings, sizes, modality_name)

    return embeddings, checksums, metadata, saved_s3_keys


def _embedding_dim(embeddings) -> int | None:
    """Vector dimension of the first non-empty embedding, else None."""
    if embeddings and hasattr(embeddings[0], "__len__"):
        return len(embeddings[0])
    return None


def _stamp_lineage(md, indexing_job_id, model_version, model_path, embedding_dim):
    """Stamp lineage fields onto one metadata dict."""
    if indexing_job_id:
        md["indexing_job_id"] = indexing_job_id
    if model_version:
        md["model_version"] = model_version
    if model_path:
        md["model_path"] = model_path
    if embedding_dim is not None:
        md["embedding_dimension"] = embedding_dim


def _inject_lineage_metadata(
    metadata, embeddings, *, indexing_job_id, model_version, model_path
):
    """Inject RAG lineage metadata (MEM-C1) into each item's metadata dict
    so store_one_embedding_in_db can persist them as column values."""
    if not (indexing_job_id or model_version or model_path):
        return
    embedding_dim = _embedding_dim(embeddings)
    for md in metadata:
        _stamp_lineage(md, indexing_job_id, model_version, model_path, embedding_dim)


def _resolve_store_fn(store_fn):
    """Default to the production SQL store when no override is supplied."""
    if store_fn is not None:
        return store_fn
    from umrs.server.sql import store_embeddings_in_db  # noqa: boundary — lazy fallback

    return store_embeddings_in_db


def compute_and_store_embedding(
    path_to_bucket,
    s3_keys,
    data_base,
    table_name,
    model,
    device,
    modality_name,
    s3_connection,
    output_bucket,
    task=None,
    skip_existing=True,
    store_fn=None,
    indexing_job_id=None,
    model_version=None,
    model_path=None,
):
    t0 = _time.monotonic()
    outcome = "ok"
    try:
        embeddings, checksums, metadata, saved_s3_keys = compute_embeddings_core(
            model,
            modality_name,
            s3_connection,
            path_to_bucket,
            s3_keys,
            output_bucket=output_bucket,
            task=task,
            skip_existing=skip_existing,
        )

        assert len(embeddings) == len(checksums)
        assert len(embeddings) == len(metadata)

        _inject_lineage_metadata(
            metadata,
            embeddings,
            indexing_job_id=indexing_job_id,
            model_version=model_version,
            model_path=model_path,
        )

        store_fn = _resolve_store_fn(store_fn)
        store_fn(
            data_base,
            embeddings,
            table_name,
            path_to_bucket,
            s3_keys,
            saved_s3_keys,
            modality_name,
            checksums,
            model.model_name,
            metadata_list=metadata,
        )
    except Exception:
        outcome = "error"
        raise
    finally:
        # X1 metrics: failure-isolated — a metric outage never breaks compute.
        try:
            METRICS.embedding_compute(
                str(modality_name), outcome, _time.monotonic() - t0
            )
        except Exception:  # noqa: BLE001
            pass


def load_embedding_model(model_name):

    if not model_name:
        print("No model...")
        return None

    # Load the sentence-transformers model
    print(f"\t\t\t--> Loading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    return model


def challenge_embeddings(model, conn, cur, query_texts):
    for query_text in query_texts:
        ids, texts = query_db_with_embeddings(model, conn, cur, query_text)
        print(f"Answers for: {query_text}:")
        for i, id in enumerate(ids):
            print(f"\t{id} -> {texts[i]}")


def query_db_with_embeddings(model, conn, cur, query_text, top_k=5):
    """
    Queries the PostgreSQL database for the most similar embeddings to the given text.

    Args:
        conn: Active PostgreSQL connection.
        cur: Cursor for executing queries.
        query_text (str): The input text for similarity search.
        top_k (int): Number of most similar results to retrieve.

    Returns:
        tuple: (ids, texts) where
            - ids (list): List of retrieved row IDs.
            - texts (list): Corresponding text values.
    """
    # Generate the embedding for the input text
    query_embedding = model.encode(query_text).tolist()

    # Perform similarity search using L2 distance
    cur.execute(
        """
        SELECT id, text
        FROM text_embeddings
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """,
        (query_embedding, top_k),
    )

    # Fetch results
    results = cur.fetchall()

    # Extract IDs and texts
    ids = [row[0] for row in results]
    texts = [row[1] for row in results]

    return ids, texts


def save_texts_in_db(model, conn, cur):

    texts = [
        "I love eating a fresh apple every morning.",  # 🍏 Apple as a fruit
        "Apple Inc. is one of the biggest tech companies in the world.",  # 💻 Apple as a company
        "The teacher gave Johnny a golden apple for his outstanding performance.",  # 🏆 Apple as a symbolic reward
        "She has a small apple tattoo on her wrist.",  # 🎨 Apple as a design
        "New York is often called the Big Apple.",  # 🏙️ Apple as a nickname for NYC
        "The apple doesn’t fall far from the tree.",  # 🧠 Apple in a proverb
        "I ordered a glass of apple cider at the bar.",  # 🍹 Apple in beverages
        "The witch in Snow White gave her a poisoned apple.",  # 📖 Apple in folklore
        "Doctors say an apple a day keeps the doctor away.",  # 🍏 Apple in a health saying
        "The software update fixed the bug in my Apple MacBook.",  # 🖥️ Apple in tech devices
        "He used to work at Apple as a software engineer.",  # 💼 Apple as a workplace
        "The apple tree in my backyard is full of ripe fruit.",  # 🌳 Apple as a tree
        "Apple pie is my favorite dessert.",  # 🥧 Apple in food
        "She dropped an apple emoji 🍎 in the text message.",  # 📱 Apple as an emoji
        "The red apple in the painting symbolizes temptation.",  # 🎨 Apple in art
        "He threw an apple across the room as a joke.",  # 🍏 Apple as a throwable object
        "We studied the gravitational effect on an apple like Newton did.",  # 🪙 Apple in physics (Newton)
        "The apple-colored dress looked amazing on her.",  # 🎨 Apple as a color
        "He gifted me a bottle of apple-scented perfume.",  # 🍏 Apple in fragrance
        "Farmers are worried about the apple harvest this season.",  # 🚜 Apple in agriculture
        "The villagers gathered under the apple tree for shade.",  # 🌳 Apple in nature
    ]

    for text in texts:
        print(f"creating embeddings for: {text}")

        # Generate embedding directly (no decoding needed)
        embedding = model.encode(text)

        print(len(embedding))

        # Insert into database
        cur.execute(
            "INSERT INTO text_embeddings (text, embedding) VALUES (%s, %s)",
            (text, embedding.tolist()),
        )
        conn.commit()

        print("Inserted embedding successfully.")


def cluster_from_embeddings(conn, cur, nb_clusters=4):
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

    # Fetch all embeddings and their corresponding texts from the database
    cur.execute("SELECT id, text, embedding FROM text_embeddings;")
    results = cur.fetchall()

    if not results:
        print("No embeddings found in the database.")
        return []

    # Extract texts and process embeddings
    texts = [row[1] for row in results]  # Extract text values
    embeddings = []

    for row in results:
        embedding_str = row[2]  # Extract the stored vector (string format)
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
        closest_idx = min(
            cluster_indices, key=lambda i: np.linalg.norm(embeddings[i] - centroid)
        )

        # Store the closest text to the centroid
        centroid_texts.append(texts[closest_idx])

    return centroid_texts


def cluster_from_embeddings_wrong(conn, cur, nb_clusters=4):
    """
    Performs K-Means clustering directly in PostgreSQL on stored embeddings
    and returns one representative text per cluster.

    Args:
        conn: Active PostgreSQL connection.
        cur: Cursor for executing queries.
        nb_clusters (int): Number of clusters.

    Returns:
        list: One representative text from each cluster.
    """

    # Ensure pgvector's kmeans function is available
    cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector';")
    if cur.fetchone()[0] == 0:
        print(
            "Error: pgvector extension is not enabled. Run 'CREATE EXTENSION vector;' in PostgreSQL."
        )
        return []

    # Perform K-Means clustering using PostgreSQL
    cur.execute(f"""
        WITH clustered AS (
            SELECT id, text, kmeans(embedding, {nb_clusters}) OVER () AS cluster_id, embedding
            FROM text_embeddings
        ),
        centroids AS (
            SELECT cluster_id, AVG(embedding) AS centroid
            FROM clustered
            GROUP BY cluster_id
        )
        SELECT clustered.text
        FROM clustered
        JOIN centroids ON clustered.cluster_id = centroids.cluster_id
        ORDER BY clustered.embedding <-> centroids.centroid
        LIMIT {nb_clusters};
    """)

    # Fetch representative texts (closest to cluster centroids)
    centroid_texts = [row[0] for row in cur.fetchall()]
    return centroid_texts


def main():
    import psycopg2

    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    dbnames = [
        "embeddings_db",
        "large_embeddings_db",
    ]  # correspinfing DB with same the correct dimensionnality

    id = 1  # 0 or 1

    model = SentenceTransformer(models[id])
    db_name = dbnames[id]

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=db_name, user="ejeannot", password="", host="localhost", port="5432"
    )
    cur = conn.cursor()

    # save_texts_in_db(model, conn, cur)

    # challenge_embeddings(model, conn, cur, texts1);

    # print("\n ------- More difficult ----------\n")

    # challenge_embeddings(model, conn, cur, texts2);
    challenge_embeddings(model, conn, cur, ["Adam and Eve, the lost paradize"])

    print("\n ------- 4 centroids texts ----------\n")

    centroid_texts = cluster_from_embeddings(conn, cur, nb_clusters=5)
    for text in centroid_texts:
        print(text)

        # Close connection
    cur.close()
    conn.close()


def open_AI_tokenizer():
    import openai

    # Set your API key
    with open("API_KEY_INRIA.txt") as api_key_file:
        openai.api_key = api_key_file.read().strip()

    # Example text to embed
    text = "Artificial intelligence is transforming the world."

    try:
        # Get embeddings from OpenAI (NEW SYNTAX)
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small",
            # model="gpt-4o-mini-2024-07-18"
        )

        # Extract embedding vector
        embedding = response.data[0].embedding

        print("Embedding vector:", embedding)
        print(
            "Embedding size:", len(embedding)
        )  # Should be 1536 for `text-embedding-ada-002`

    except openai.OpenAIError as e:
        # Pretty-print error as JSON if it's a dictionary
        try:
            error_data = e.response.json()
            print("OpenAI API error:")
            print(json.dumps(error_data, indent=4))
        except AttributeError:
            print(f"OpenAI API error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")
