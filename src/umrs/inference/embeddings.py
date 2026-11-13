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
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import ast  # To safely convert strings to lists
import os
import time as _time

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from umrs.metrics import METRICS
from umrs.shared.env_knobs import env_int
from umrs.shared.storage.s3 import load_data_from_s3
from umrs.shared.utils import zero_out_embeddings_for_empty_objects

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Module logger — the batch path (compute_and_store_embedding_batch) logs
# per-item failures; upstream references `logger` without defining it
# (latent NameError on the error path).
logger = logging.getLogger(__name__)

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


def is_direct_compute(model, modality_name, task) -> bool:
    """True when the model does its own S3 I/O (no pre-loaded objects)."""
    model_type = getattr(model, "type", None)
    return (
        model_type in DIRECT_COMPUTE_MODEL_TYPES
        or (task is not None and task in DIRECT_COMPUTE_TASKS)
        or (task is None and modality_name in DIRECT_COMPUTE_MODALITIES)
    )


def inference_batch_size() -> int:
    """``UMRS_INFERENCE_BATCH_SIZE``: max objects per model forward pass.

    0 (the default) keeps the single forward pass over the whole batch.
    A value that is not a non-negative integer falls back to 0 with a
    warning (``env_knobs`` rule), so a typo never stops ingestion.
    """
    return env_int("UMRS_INFERENCE_BATCH_SIZE", 0, lo=0)


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
    if is_direct_compute(model, modality_name, task):
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
        except Exception:  # noqa: BLE001, S110
            pass


def compute_and_store_embedding_batch(items, store_fn=None):
    """Batch variant: one model across multiple (bucket, keys) items.

    Each item in `items` is a dict with keys:
        path_to_bucket, s3_keys, data_base, table_name, model, device,
        modality_name, s3_connection, output_bucket, task, skip_existing,
        indexing_job_id, model_version, model_path

    All items must share the same model. Returns a list of per-item exit bools
    (True = success). With ``UMRS_INFERENCE_BATCH_SIZE`` > 0 each forward
    pass holds at most that many objects and the S3 load of the next batch
    overlaps the compute of the current one (``_pipelined_embed``).
    """
    if not items:
        return []

    model = items[0]["model"]
    if is_direct_compute(model, items[0]["modality_name"], items[0].get("task")):
        return _process_direct_batch(items, store_fn)

    batch_size = inference_batch_size()
    t_wall0 = _time.perf_counter()
    per_item, t_io, t_compute = _embed_items(items, model, batch_size)
    t_wall = _time.perf_counter() - t_wall0

    n_objects = sum(len(data.checksums) for data in per_item)
    if n_objects == 0:
        return [False] * len(items)
    _log_embed_timing(model, n_objects, batch_size, t_wall, t_io, t_compute)

    # Route through the sanctioned fallback (single boundary-tagged point).
    store_fn = _resolve_store_fn(store_fn)
    return [
        _store_batch_slice(
            store_fn, item, data.embeddings, data.checksums, data.metadata,
            [None] * len(item["s3_keys"]), data.sizes, data.has_zero,
            model.model_name,
        )
        for item, data in zip(items, per_item)
    ]


def _log_embed_timing(model, n_objects, batch_size, t_wall, t_io, t_compute):
    """DEBUG timing line; ``umrs_embed_timing`` carries the same numbers."""
    logger.debug(
        "[embed] n=%d wall=%.3fs io=%.3fs compute=%.3fs model=%s batch_size=%d",
        n_objects, t_wall, t_io, t_compute,
        getattr(model, "model_name", "?"), batch_size,
        extra={"umrs_embed_timing": {
            "n_objects": n_objects, "wall_s": t_wall, "io_s": t_io,
            "compute_s": t_compute, "batch_size": batch_size,
        }},
    )


def _process_direct_batch(items, store_fn):
    """Direct-compute models manage their own I/O; process per-item."""
    results = []
    for item in items:
        try:
            compute_and_store_embedding(
                item["path_to_bucket"], item["s3_keys"], item["data_base"],
                item["table_name"], item["model"], item["device"],
                item["modality_name"], item["s3_connection"], item["output_bucket"],
                task=item.get("task"), skip_existing=item.get("skip_existing", True),
                store_fn=store_fn,
            )
            results.append(True)
        except Exception:
            logger.exception("compute_and_store_embedding_batch: item failed")
            results.append(False)
    return results


# ---------------------------------------------------------------------------
# Standard-path compute: sequential (one pass) or pipelined (capped passes).
# ---------------------------------------------------------------------------

# Batches loaded ahead of the one being computed (bounds memory).
_PREFETCH_DEPTH = 2


class _Segment(NamedTuple):
    """Consecutive keys of one item, loaded from S3 in one call."""

    item_idx: int
    objects: list
    metadata: list
    checksums: list
    sizes: list
    has_zero: bool


@dataclass
class _ItemData:
    """Everything one item needs for its store call, in key order."""

    embeddings: list = field(default_factory=list)
    checksums: list = field(default_factory=list)
    metadata: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    has_zero: bool = False

    def add(self, segment: _Segment, embeddings) -> None:
        self.embeddings.extend(embeddings)
        self.checksums.extend(segment.checksums)
        self.metadata.extend(segment.metadata)
        self.sizes.extend(segment.sizes)
        self.has_zero = self.has_zero or segment.has_zero


def _load_segment(items, item_idx, keys) -> _Segment:
    item = items[item_idx]
    loaded = load_data_from_s3(
        item["s3_connection"], item["path_to_bucket"], keys, item["modality_name"],
    )
    return _Segment(item_idx, *loaded)


def _distribute(per_item, segments, embeddings) -> None:
    """Hand each segment its rows of ``embeddings`` (same order as loaded)."""
    offset = 0
    for segment in segments:
        count = len(segment.objects)
        per_item[segment.item_idx].add(segment, embeddings[offset:offset + count])
        offset += count


def _compute(model, segments):
    """One forward pass over the segments' objects -> (embeddings, seconds)."""
    objects = [obj for segment in segments for obj in segment.objects]
    if not objects:
        return [], 0.0
    t0 = _time.perf_counter()
    embeddings = model.compute_embeddings(objects)
    return embeddings, _time.perf_counter() - t0


def _embed_items(items, model, batch_size):
    """Returns ``(per_item, io_seconds, compute_seconds)``."""
    if batch_size > 0:
        return _pipelined_embed(items, model, batch_size)
    return _sequential_embed(items, model)


def _sequential_embed(items, model):
    """Load every item, then one forward pass over all objects."""
    t0 = _time.perf_counter()
    segments = [
        _load_segment(items, idx, item["s3_keys"]) for idx, item in enumerate(items)
    ]
    t_io = _time.perf_counter() - t0
    embeddings, t_compute = _compute(model, segments)
    per_item = [_ItemData() for _ in items]
    _distribute(per_item, segments, embeddings)
    return per_item, t_io, t_compute


def _iter_key_batches(items, batch_size):
    """Flatten every item's keys, then cut them into ``batch_size`` slices."""
    flat = [(idx, key) for idx, item in enumerate(items) for key in item["s3_keys"]]
    for start in range(0, len(flat), batch_size):
        yield flat[start:start + batch_size]


def _group_runs(key_batch):
    """``[(idx, key), ...]`` -> ``[(idx, [keys])]`` for consecutive equal idx."""
    runs: list[tuple[int, list]] = []
    for idx, key in key_batch:
        if runs and runs[-1][0] == idx:
            runs[-1][1].append(key)
        else:
            runs.append((idx, [key]))
    return runs


def _load_key_batch(items, key_batch):
    return [_load_segment(items, idx, keys) for idx, keys in _group_runs(key_batch)]


class _PrefetchError:
    """Carries a loader exception from the prefetch thread to the consumer."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _Prefetcher:
    """Load batches on one background thread, at most ``depth`` ahead.

    Iterating yields loaded batches in order and re-raises a loader
    failure on the consumer thread. Leaving the ``with`` block (normally
    or on an exception) stops the loader and joins it, so no thread
    outlives the call.
    """

    _DONE = object()

    def __init__(self, load, batches, depth=_PREFETCH_DEPTH):
        self._load = load
        self._batches = batches
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self.io_seconds = 0.0
        self._thread = threading.Thread(
            target=self._run, name="umrs-embed-prefetch", daemon=True,
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._drain()
        self._thread.join()
        return False

    def __iter__(self):
        while True:
            loaded = self._queue.get()
            if loaded is self._DONE:
                return
            if isinstance(loaded, _PrefetchError):
                raise loaded.exc
            yield loaded

    def _run(self) -> None:
        try:
            for batch in self._batches:
                t0 = _time.perf_counter()
                loaded = self._load(batch)
                self.io_seconds += _time.perf_counter() - t0
                if not self._put(loaded):
                    return
        except Exception as exc:  # noqa: BLE001 — re-raised on the consumer thread
            self._put(_PrefetchError(exc))
            return
        self._put(self._DONE)

    def _put(self, loaded) -> bool:
        """Blocking put that gives up once the consumer has left."""
        while not self._stop.is_set():
            try:
                self._queue.put(loaded, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


def _pipelined_embed(items, model, batch_size):
    """Overlap the S3 load of batch N+1 with the forward pass of batch N."""
    per_item = [_ItemData() for _ in items]
    t_compute = 0.0
    batches = list(_iter_key_batches(items, batch_size))
    with _Prefetcher(lambda batch: _load_key_batch(items, batch), batches) as prefetch:
        for segments in prefetch:
            embeddings, seconds = _compute(model, segments)
            t_compute += seconds
            _distribute(per_item, segments, embeddings)
    return per_item, prefetch.io_seconds, t_compute


def _job_metadata_fields(idj, mv, mp, embeddings) -> dict:
    """Concrete job fields for one item (only present keys, + dimension)."""
    fields = {}
    if idj:
        fields["indexing_job_id"] = idj
    if mv:
        fields["model_version"] = mv
    if mp:
        fields["model_path"] = mp
    dim = len(embeddings[0]) if embeddings and hasattr(embeddings[0], "__len__") else None
    if dim is not None:
        fields["embedding_dimension"] = dim
    return fields


def _stamp_job_metadata(metadata, item, embeddings) -> None:
    """indexing_job_id / model_version / model_path (+dimension) onto each row."""
    idj = item.get("indexing_job_id")
    mv = item.get("model_version")
    mp = item.get("model_path")
    if not (idj or mv or mp):
        return
    extra = _job_metadata_fields(idj, mv, mp, embeddings)
    for md in metadata:
        md.update(extra)


def _store_batch_slice(store_fn, item, embeddings, checksums, metadata, saved,
                       sizes, has_zero, model_name) -> bool:
    """Zero-stamp when needed, stamp job metadata, store one item's slice."""
    if has_zero:
        zero_out_embeddings_for_empty_objects(embeddings, sizes, item["modality_name"])
    _stamp_job_metadata(metadata, item, embeddings)
    try:
        store_fn(
            item["data_base"], embeddings, item["table_name"],
            item["path_to_bucket"], item["s3_keys"], saved,
            item["modality_name"], checksums, model_name,
            metadata_list=metadata,
        )
        return True
    except Exception:
        logger.exception("compute_and_store_embedding_batch: store failed")
        return False


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

    except Exception as e:  # noqa: BLE001 — unknown OpenAI error shape prints, never raises
        print(f"Unexpected error: {e}")
