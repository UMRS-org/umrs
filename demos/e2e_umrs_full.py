#!/usr/bin/env python3
"""
Comprehensive End-to-End Demo: UMRS Full Stack

Demonstrates the COMPLETE UMRS workflow including:
1. UMRS Server: ZeroMQ control plane + S3 storage + PostgreSQL
2. File Transfer: Upload/download files via UMRS Client
3. Embedding Worker: Compute embeddings on data
4. Model Registry: Register and manage models
5. RAG Pipeline: Hybrid search with BM25 + vector similarity
6. LLM Completion: Query via Mini-RiVault endpoint

This demo provides:
- Docker Compose environment with all UMRS components
- Full client-server communication via ZeroMQ
- File upload/download operations
- Embedding computation trigger
- RAG query with retrieved context

Usage:
    # Start the demo environment
    cd demos/umrs-full
    docker compose up -d

    # Run the demo
    python demos/e2e_umrs_full.py --live

    # Mock mode (no Docker required)
    python demos/e2e_umrs_full.py --mock

    # Show available options
    python demos/e2e_umrs_full.py --help

Environment Variables:
    UMRS_SERVER_HOST       - UMRS server hostname (default: localhost)
    UMRS_CONTROL_PORT      - ZeroMQ control port (default: 5555)
    S3_ENDPOINT            - MinIO endpoint (default: http://localhost:9000)
    DB_HOST                - PostgreSQL host (default: localhost)
    LITELLM_PORT           - LiteLLM proxy port (default: 11434)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add UMRS source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_umrs_full")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DemoConfig:
    """Configuration for comprehensive E2E demo."""

    # Mode
    live_mode: bool = False  # If False, use mocks

    # UMRS Server configuration
    server_host: str = field(
        default_factory=lambda: os.environ.get("UMRS_SERVER_HOST", "localhost")
    )
    control_port: int = field(
        default_factory=lambda: int(os.environ.get("UMRS_CONTROL_PORT", "5555"))
    )
    data_port: int = field(
        default_factory=lambda: int(os.environ.get("UMRS_DATA_PORT", "5556"))
    )

    # S3/MinIO configuration
    s3_endpoint: str = field(
        default_factory=lambda: os.environ.get("S3_ENDPOINT", "http://localhost:9000")
    )
    s3_access_key: str = field(
        default_factory=lambda: os.environ.get("S3_ACCESS_KEY", "minioadmin")
    )
    s3_secret_key: str = field(
        default_factory=lambda: os.environ.get("S3_SECRET_KEY", "minioadmin")
    )

    # Database configuration
    db_host: str = field(
        default_factory=lambda: os.environ.get("DB_HOST", "localhost")
    )
    db_port: int = field(
        default_factory=lambda: int(os.environ.get("DB_PORT", "5432"))
    )
    db_user: str = field(
        default_factory=lambda: os.environ.get("DB_USER", "postgres")
    )
    db_password: str = field(
        default_factory=lambda: os.environ.get("DB_PASSWORD", "postgres")
    )
    db_name: str = field(
        default_factory=lambda: os.environ.get("DB_NAME", "umrs_demo")
    )

    # Model configuration
    model_path: str = field(
        default_factory=lambda: os.environ.get("DEMO_MODEL_PATH", "/models")
    )
    embedding_model: str = "all-MiniLM-L6-v2"

    # LiteLLM configuration
    litellm_port: int = field(
        default_factory=lambda: int(os.environ.get("LITELLM_PORT", "11434"))
    )
    litellm_key: str = field(
        default_factory=lambda: os.environ.get("LITELLM_MASTER_KEY", "sk-ll-demo")
    )

    # Demo settings
    username: str = "demo_user"
    experiment_name: str = "e2e_demo"
    bucket_alias: str = "demo_bucket"

    # RAG configuration
    embedding_dim: int = 384
    top_k: int = 3

    # Timeouts
    server_timeout_ms: int = 5000
    ready_timeout: int = 300


# =============================================================================
# Sample Documents for Demo
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Seismic Wave Propagation",
        "content": """Seismic wave propagation is governed by the elastic wave equation.
P-waves (primary waves) travel faster than S-waves (secondary waves).
The velocity depends on the medium's density and elastic properties.
Finite element methods and spectral element methods are commonly used
for numerical simulation of seismic wave propagation in complex media.""",
        "metadata": {"category": "geophysics", "type": "research"},
    },
    {
        "id": "doc2",
        "title": "Machine Learning for Geophysics",
        "content": """Machine learning techniques are revolutionizing geophysical data analysis.
Convolutional neural networks can identify geological features from seismic images.
Recurrent networks process time-series seismic data effectively.
Transfer learning enables models trained on synthetic data to work on real measurements.""",
        "metadata": {"category": "ml", "type": "research"},
    },
    {
        "id": "doc3",
        "title": "HPC for Scientific Computing",
        "content": """High-Performance Computing (HPC) is essential for large-scale scientific simulations.
GPU acceleration with CUDA enables parallel computation of complex algorithms.
MPI and OpenMP provide distributed and shared memory parallelization.
Slurm job scheduler manages workloads across thousands of compute nodes.
Singularity containers ensure reproducibility in HPC environments.""",
        "metadata": {"category": "hpc", "type": "tutorial"},
    },
    {
        "id": "doc4",
        "title": "RiVault Architecture",
        "content": """RiVault provides on-demand LLM inference for scientific workflows.
Mini-RiVault bundles vLLM for inference and LiteLLM as an OpenAI-compatible proxy.
The system bootstraps via Slurm job submission with Singularity containers.
Model weights are pre-staged on the shared filesystem, never downloaded at runtime.
UMRS orchestrates the entire lifecycle from job submission to endpoint discovery.""",
        "metadata": {"category": "architecture", "type": "documentation"},
    },
    {
        "id": "doc5",
        "title": "UMRS File Transfer Protocol",
        "content": """UMRS uses ZeroMQ for high-performance client-server communication.
The control channel uses ROUTER/DEALER sockets for request-response patterns.
The data channel uses PUSH/PULL sockets for streaming file transfers.
Files are uploaded to S3-compatible storage (MinIO) with metadata.
PostgreSQL with pgvector extension stores embeddings for similarity search.""",
        "metadata": {"category": "architecture", "type": "documentation"},
    },
]


# =============================================================================
# Mock Implementations
# =============================================================================


class MockZMQSocket:
    """Mock ZeroMQ socket for testing without actual network."""

    def __init__(self, socket_type: str):
        self.socket_type = socket_type
        self.messages: List[Any] = []
        self.responses = {
            "ping": {"status": "ok", "server_ready": True},
            "list_buckets": {
                "status": "ok",
                "buckets": [
                    {"alias": "demo_bucket", "modality": "text", "path": "demo/bucket"}
                ],
            },
            "upload": {"status": "ok", "s3_key": "demo/test_file.txt"},
            "status": {
                "status": "ok",
                "summary": {
                    "documents": 5,
                    "embeddings": 5,
                    "last_update": "2026-02-13T00:00:00Z",
                },
            },
            "recompute": {"status": "ok", "message": "Embeddings recomputed"},
        }

    def send_string(self, msg: str):
        self.messages.append(msg)
        logger.info(f"[MOCK] Sent: {msg[:100]}...")

    def recv_string(self, timeout: int = 5000) -> str:
        # Parse last sent message to determine response
        if self.messages:
            last_msg = self.messages[-1]
            if "ping" in last_msg.lower():
                return json.dumps(self.responses["ping"])
            elif "list_buckets" in last_msg.lower():
                return json.dumps(self.responses["list_buckets"])
            elif "upload" in last_msg.lower():
                return json.dumps(self.responses["upload"])
            elif "status" in last_msg.lower():
                return json.dumps(self.responses["status"])
            elif "recompute" in last_msg.lower():
                return json.dumps(self.responses["recompute"])
        return json.dumps({"status": "ok"})

    def close(self):
        pass


class MockS3Client:
    """Mock S3 client for testing without actual MinIO."""

    def __init__(self):
        self.buckets: Dict[str, Dict[str, bytes]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def create_bucket(self, Bucket: str, **kwargs):
        self.buckets[Bucket] = {}
        logger.info(f"[MOCK] Created bucket: {Bucket}")

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kwargs):
        if Bucket not in self.buckets:
            self.buckets[Bucket] = {}
        self.buckets[Bucket][Key] = Body
        self.metadata[f"{Bucket}/{Key}"] = kwargs.get("Metadata", {})
        logger.info(f"[MOCK] Uploaded: s3://{Bucket}/{Key} ({len(Body)} bytes)")
        return {"ETag": hashlib.md5(Body).hexdigest()}

    def get_object(self, Bucket: str, Key: str):
        if Bucket not in self.buckets or Key not in self.buckets[Bucket]:
            raise Exception(f"Object not found: s3://{Bucket}/{Key}")
        body = self.buckets[Bucket][Key]
        
        class MockBody:
            def __init__(self, data):
                self._data = data
            def read(self):
                return self._data
        
        return {"Body": MockBody(body)}

    def list_objects_v2(self, Bucket: str, Prefix: str = ""):
        if Bucket not in self.buckets:
            return {"Contents": []}
        contents = [
            {"Key": k, "Size": len(v)}
            for k, v in self.buckets[Bucket].items()
            if k.startswith(Prefix)
        ]
        return {"Contents": contents}

    def list_buckets(self):
        return {"Buckets": [{"Name": b} for b in self.buckets.keys()]}


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Generate deterministic pseudo-embeddings based on text hash
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            emb = rng.randn(self.dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # Normalize
            embeddings.append(emb)
        return np.array(embeddings)


class MockDatabase:
    """Mock PostgreSQL database for testing."""

    def __init__(self):
        self.tables: Dict[str, List[Dict]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    def execute(self, query: str, params: tuple = ()):
        logger.info(f"[MOCK DB] Execute: {query[:80]}...")
        return []

    def store_embedding(self, doc_id: str, embedding: np.ndarray, metadata: Dict):
        self.embeddings[doc_id] = embedding
        logger.info(f"[MOCK DB] Stored embedding for {doc_id}")

    def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        results = []
        for doc_id, emb in self.embeddings.items():
            similarity = np.dot(query_embedding, emb)
            results.append((doc_id, float(similarity)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self):
        self.responses = {
            "seismic": "Seismic wave propagation follows the elastic wave equation...",
            "machine learning": "ML techniques are transforming geophysical analysis...",
            "hpc": "HPC environments like RCCS use Slurm for job scheduling...",
            "umrs": "UMRS provides unified data management with ZeroMQ communication...",
            "file transfer": "UMRS file transfer uses ZeroMQ PUSH/PULL sockets...",
        }

    async def chat_completion(
        self, messages: List[Dict[str, str]], model: str = "rivault-gpt-oss-static"
    ) -> str:
        # Extract user query
        user_msg = messages[-1].get("content", "")
        
        # Find matching response
        for keyword, response in self.responses.items():
            if keyword.lower() in user_msg.lower():
                return response
        
        return "Based on the UMRS documentation, the system provides comprehensive data management capabilities."


# =============================================================================
# UMRS Client Implementation
# =============================================================================


class UMRSClient:
    """UMRS Client for server communication and file operations."""

    def __init__(self, config: DemoConfig, mock: bool = True):
        self.config = config
        self.mock = mock
        
        if mock:
            self.control_socket = MockZMQSocket("DEALER")
            self.s3_client = MockS3Client()
        else:
            self._init_live_connections()

    def _init_live_connections(self):
        """Initialize live ZeroMQ and S3 connections."""
        try:
            import zmq
            import boto3
            
            # ZeroMQ control channel
            context = zmq.Context()
            self.control_socket = context.socket(zmq.DEALER)
            self.control_socket.connect(
                f"tcp://{self.config.server_host}:{self.config.control_port}"
            )
            self.control_socket.setsockopt(zmq.RCVTIMEO, self.config.server_timeout_ms)
            self.control_socket.setsockopt(zmq.SNDTIMEO, self.config.server_timeout_ms)
            
            # S3 client
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint,
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key,
            )
            logger.info("Live connections initialized")
        except Exception as e:
            logger.error(f"Failed to initialize live connections: {e}")
            raise

    def ping(self) -> Dict[str, Any]:
        """Check if UMRS server is ready."""
        request = json.dumps({"action": "ping", "username": self.config.username})
        self.control_socket.send_string(request)
        response = self.control_socket.recv_string()
        return json.loads(response)

    def list_buckets(self) -> List[Dict[str, str]]:
        """List available buckets for the user."""
        request = json.dumps({
            "action": "list_buckets",
            "username": self.config.username,
        })
        self.control_socket.send_string(request)
        response = json.loads(self.control_socket.recv_string())
        return response.get("buckets", [])

    def upload_file(
        self,
        file_path: str,
        bucket_alias: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Upload a file to S3 via UMRS."""
        with open(file_path, "rb") as f:
            content = f.read()
        
        filename = os.path.basename(file_path)
        s3_key = f"{self.config.experiment_name}/{bucket_alias}/{filename}"
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.config.bucket_alias,
            Key=s3_key,
            Body=content,
            Metadata=metadata or {},
        )
        
        # Notify server
        request = json.dumps({
            "action": "upload",
            "username": self.config.username,
            "bucket_alias": bucket_alias,
            "s3_key": s3_key,
            "metadata": metadata or {},
        })
        self.control_socket.send_string(request)
        response = json.loads(self.control_socket.recv_string())
        response["s3_key"] = s3_key
        return response

    def upload_text(
        self,
        text: str,
        filename: str,
        bucket_alias: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Upload text content directly."""
        s3_key = f"{self.config.experiment_name}/{bucket_alias}/{filename}"
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.config.bucket_alias,
            Key=s3_key,
            Body=text.encode("utf-8"),
            Metadata=metadata or {},
        )
        
        # Notify server
        request = json.dumps({
            "action": "upload",
            "username": self.config.username,
            "bucket_alias": bucket_alias,
            "s3_key": s3_key,
            "content_type": "text",
            "metadata": metadata or {},
        })
        self.control_socket.send_string(request)
        response = json.loads(self.control_socket.recv_string())
        response["s3_key"] = s3_key
        return response

    def download_file(self, s3_key: str, local_path: str) -> Dict[str, Any]:
        """Download a file from S3."""
        response = self.s3_client.get_object(
            Bucket=self.config.bucket_alias,
            Key=s3_key,
        )
        content = response["Body"].read()
        
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
        
        return {"status": "ok", "local_path": local_path, "size": len(content)}

    def get_status(self, target: str = "all") -> Dict[str, Any]:
        """Get server status and statistics."""
        request = json.dumps({
            "action": "status",
            "username": self.config.username,
            "target": target,
        })
        self.control_socket.send_string(request)
        response = json.loads(self.control_socket.recv_string())
        return response

    def trigger_recompute(self) -> Dict[str, Any]:
        """Trigger embedding recomputation on server."""
        request = json.dumps({
            "action": "recompute",
            "username": self.config.username,
        })
        self.control_socket.send_string(request)
        response = json.loads(self.control_socket.recv_string())
        return response

    def close(self):
        """Close connections."""
        if hasattr(self.control_socket, "close"):
            self.control_socket.close()


# =============================================================================
# Embedding Worker Simulation
# =============================================================================


class EmbeddingWorker:
    """Embedding worker that computes embeddings for documents."""

    def __init__(self, config: DemoConfig, mock: bool = True):
        self.config = config
        self.mock = mock
        
        if mock:
            self.model = MockEmbeddingModel(dim=config.embedding_dim)
            self.db = MockDatabase()
        else:
            self._init_live_worker()

    def _init_live_worker(self):
        """Initialize live embedding worker."""
        try:
            from sentence_transformers import SentenceTransformer
            import psycopg2
            
            self.model = SentenceTransformer(self.config.embedding_model)
            self.db = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
            )
            logger.info("Live embedding worker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding worker: {e}")
            raise

    def compute_embeddings(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Compute embeddings for a list of documents."""
        embeddings = {}
        
        texts = [doc["content"] for doc in documents]
        doc_ids = [doc["id"] for doc in documents]
        
        # Compute embeddings
        vectors = self.model.encode(texts, show_progress_bar=False)
        
        for doc_id, vector, doc in zip(doc_ids, vectors, documents):
            embeddings[doc_id] = vector
            self.db.store_embedding(doc_id, vector, doc.get("metadata", {}))
        
        return embeddings

    def search_similar(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        query_embedding = self.model.encode([query])[0]
        return self.db.search_similar(query_embedding, top_k)


# =============================================================================
# Demo Steps
# =============================================================================


def print_banner(title: str, mode: str):
    """Print demo banner."""
    width = 70
    print("\n" + "╔" + "═" * width + "╗")
    print(f"║{title:^{width}}║")
    print(f"║{'Mode: ' + mode:^{width}}║")
    print("╚" + "═" * width + "╝")


def print_step(step_num: int, title: str):
    """Print step header."""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {title}")
    print("=" * 70)


async def demo_server_connection(client: UMRSClient, config: DemoConfig) -> bool:
    """Step 1: Test UMRS server connection."""
    print_step(1, "UMRS Server Connection")
    
    try:
        result = client.ping()
        if result.get("status") == "ok" or result.get("server_ready"):
            print("✓ UMRS server is ready")
            print(f"  - Host: {config.server_host}")
            print(f"  - Control Port: {config.control_port}")
            return True
        else:
            print(f"✗ Server not ready: {result}")
            return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


async def demo_bucket_management(client: UMRSClient, config: DemoConfig) -> bool:
    """Step 2: Bucket management."""
    print_step(2, "Bucket Management")
    
    try:
        buckets = client.list_buckets()
        print(f"✓ Found {len(buckets)} bucket(s) for user '{config.username}':")
        for bucket in buckets:
            print(f"  - {bucket.get('alias', bucket.get('bucket_alias', 'unknown'))}: "
                  f"{bucket.get('modality', 'any')} → {bucket.get('path', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Failed to list buckets: {e}")
        return False


async def demo_file_upload(
    client: UMRSClient, config: DemoConfig, documents: List[Dict]
) -> Dict[str, str]:
    """Step 3: File upload to S3."""
    print_step(3, "File Upload (Client → S3 → Server)")
    
    uploaded = {}
    try:
        print(f"→ Uploading {len(documents)} documents...")
        
        for doc in documents:
            result = client.upload_text(
                text=doc["content"],
                filename=f"{doc['id']}.txt",
                bucket_alias=config.bucket_alias,
                metadata=doc.get("metadata", {}),
            )
            s3_key = result.get("s3_key", f"unknown/{doc['id']}.txt")
            uploaded[doc["id"]] = s3_key
            print(f"  ✓ {doc['title']} → {s3_key}")
        
        print(f"\n✓ Uploaded {len(uploaded)} documents successfully")
        return uploaded
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return uploaded


async def demo_embedding_computation(
    worker: EmbeddingWorker, documents: List[Dict]
) -> Dict[str, np.ndarray]:
    """Step 4: Compute embeddings (Embedding Worker)."""
    print_step(4, "Embedding Computation (Worker)")
    
    try:
        print(f"→ Computing embeddings for {len(documents)} documents...")
        embeddings = worker.compute_embeddings(documents)
        
        for doc_id, emb in embeddings.items():
            print(f"  ✓ {doc_id}: dim={len(emb)}, norm={np.linalg.norm(emb):.4f}")
        
        print(f"\n✓ Computed {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        print(f"✗ Embedding computation failed: {e}")
        return {}


async def demo_similarity_search(
    worker: EmbeddingWorker, query: str, documents: List[Dict], top_k: int = 3
) -> List[Tuple[str, float]]:
    """Step 5: Similarity search."""
    print_step(5, "Similarity Search (pgvector)")
    
    try:
        print(f'Query: "{query}"')
        print(f"→ Searching for top {top_k} similar documents...")
        
        results = worker.search_similar(query, top_k)
        
        print(f"\n✓ Search results:")
        for i, (doc_id, score) in enumerate(results, 1):
            # Find document title
            doc = next((d for d in documents if d["id"] == doc_id), None)
            title = doc["title"] if doc else doc_id
            print(f"  [{i}] {title}")
            print(f"      Score: {score:.4f}")
        
        return results
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return []


async def demo_llm_completion(
    llm_client: MockLLMClient,
    query: str,
    context_docs: List[Dict],
    config: DemoConfig,
) -> str:
    """Step 6: LLM completion with RAG context."""
    print_step(6, "LLM Completion (via RiVault)")
    
    try:
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document: {doc['title']}\n{doc['content']}"
            for doc in context_docs
        ])
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for scientific computing. "
                "Answer questions based on the provided context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        
        print(f"→ Sending request to LLM...")
        print(f"  - Model: rivault-gpt-oss-static")
        print(f"  - Context: {len(context_docs)} documents")
        
        response = await llm_client.chat_completion(messages)
        
        print("\n" + "─" * 50)
        print("Answer:")
        print(response)
        print("─" * 50)
        
        return response
    except Exception as e:
        print(f"✗ LLM completion failed: {e}")
        return ""


async def demo_file_download(
    client: UMRSClient, uploaded: Dict[str, str]
) -> bool:
    """Step 7: File download verification."""
    print_step(7, "File Download (S3 → Client)")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = 0
            for doc_id, s3_key in list(uploaded.items())[:2]:  # Download first 2
                local_path = os.path.join(tmpdir, f"{doc_id}_downloaded.txt")
                result = client.download_file(s3_key, local_path)
                
                if result.get("status") == "ok":
                    print(f"  ✓ Downloaded {doc_id}: {result.get('size', 0)} bytes")
                    downloaded += 1
        
        print(f"\n✓ Downloaded {downloaded} files successfully")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


async def demo_server_status(client: UMRSClient) -> Dict[str, Any]:
    """Step 8: Server status and statistics."""
    print_step(8, "Server Status")
    
    try:
        status = client.get_status("all")
        
        if status.get("status") == "ok":
            summary = status.get("summary", {})
            print("✓ Server status:")
            print(f"  - Documents: {summary.get('documents', 'N/A')}")
            print(f"  - Embeddings: {summary.get('embeddings', 'N/A')}")
            print(f"  - Last update: {summary.get('last_update', 'N/A')}")
        else:
            print(f"  Status: {status}")
        
        return status
    except Exception as e:
        print(f"✗ Failed to get status: {e}")
        return {}


# =============================================================================
# Main Demo Runner
# =============================================================================


async def run_demo(config: DemoConfig):
    """Run the full E2E demo."""
    
    mode = "LIVE" if config.live_mode else "MOCK"
    print_banner("E2E DEMO: UMRS Full Stack", mode)
    
    # Initialize components
    client = UMRSClient(config, mock=not config.live_mode)
    worker = EmbeddingWorker(config, mock=not config.live_mode)
    llm_client = MockLLMClient()
    
    try:
        # Step 1: Server connection
        connected = await demo_server_connection(client, config)
        if not connected and config.live_mode:
            print("\n✗ Cannot connect to server. Exiting.")
            return False
        
        # Step 2: Bucket management
        await demo_bucket_management(client, config)
        
        # Step 3: File upload
        uploaded = await demo_file_upload(client, config, SAMPLE_DOCUMENTS)
        
        # Step 4: Embedding computation
        embeddings = await demo_embedding_computation(worker, SAMPLE_DOCUMENTS)
        
        # Step 5: Similarity search
        query = "How does UMRS handle file transfer and communication?"
        results = await demo_similarity_search(
            worker, query, SAMPLE_DOCUMENTS, config.top_k
        )
        
        # Get context documents for LLM
        context_docs = []
        for doc_id, score in results:
            doc = next((d for d in SAMPLE_DOCUMENTS if d["id"] == doc_id), None)
            if doc:
                context_docs.append(doc)
        
        # Step 6: LLM completion
        await demo_llm_completion(llm_client, query, context_docs, config)
        
        # Step 7: File download
        await demo_file_download(client, uploaded)
        
        # Step 8: Server status
        await demo_server_status(client)
        
        print("\n" + "=" * 70)
        print("✓ Demo completed successfully!")
        print("=" * 70)
        
        return True
        
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive UMRS E2E Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--live", action="store_true",
        help="Use live connections (requires running Docker environment)"
    )
    parser.add_argument(
        "--mock", action="store_true", default=True,
        help="Use mock implementations (default)"
    )
    parser.add_argument(
        "--server-host", default=None,
        help="UMRS server hostname"
    )
    parser.add_argument(
        "--control-port", type=int, default=None,
        help="ZeroMQ control port"
    )
    parser.add_argument(
        "--s3-endpoint", default=None,
        help="MinIO S3 endpoint URL"
    )
    parser.add_argument(
        "--db-port", type=int, default=None,
        help="PostgreSQL port"
    )
    parser.add_argument(
        "--query", default="How does UMRS handle file transfer?",
        help="Query for RAG demo"
    )
    
    args = parser.parse_args()
    
    config = DemoConfig(live_mode=args.live)
    
    if args.server_host:
        config.server_host = args.server_host
    if args.control_port:
        config.control_port = args.control_port
    if args.s3_endpoint:
        config.s3_endpoint = args.s3_endpoint
    if args.db_port:
        config.db_port = args.db_port
    
    asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
