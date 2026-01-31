# UMRS Testing Guide

This document provides detailed testing guidelines specifically for the UMRS (Unified Metadata & RAG Service) component.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [API Testing](#api-testing)
6. [Database Testing](#database-testing)
7. [Running Tests](#running-tests)
8. [Test Fixtures](#test-fixtures)
9. [Mocking Guidelines](#mocking-guidelines)
10. [TDD Examples](#tdd-examples)

---

## Overview

UMRS consists of several interconnected modules:

```
┌─────────────────────────────────────────────────────────┐
│                        UMRS                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Server    │  │   Client    │  │  Message    │     │
│  │  ZMQ-based  │  │   Library   │  │   Queue     │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│  ┌──────▼────────────────▼────────────────▼──────┐     │
│  │               Core Operations                  │     │
│  │  - S3 operations (upload, download, list)     │     │
│  │  - PostgreSQL (metadata, embeddings)          │     │
│  │  - File processing                            │     │
│  └───────────────────────────────────────────────┘     │
│  ┌─────────────┐  ┌─────────────┐                       │
│  │  Embedding  │  │   RiVault   │                       │
│  │   Worker    │  │   Service   │                       │
│  └─────────────┘  └─────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

Each module requires specific testing strategies.

---

## Test Structure

```
umrs/
├── src/
│   └── umrs/
│       ├── __init__.py
│       ├── MessageQueue.py      # ← Unit tests needed
│       ├── embedding_worker.py  # ← Unit tests needed
│       ├── server_umrs.py       # ← Integration tests
│       └── client_umrs.py       # ← Unit tests needed
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_message_queue.py    # MessageQueue unit tests
│   ├── test_client.py           # Client unit tests
│   ├── test_embedding_worker.py # Embedding worker tests
│   ├── test_server.py           # Server integration tests
│   ├── test_embeddings/         # Embedding-specific tests
│   ├── minio_trigger_test/      # MinIO event tests
│   └── s3_test/                 # S3 operation tests
└── docker/
    ├── test_umrs_api.sh         # API integration tests
    └── test_rag_end_to_end.sh   # Full E2E tests
```

---

## Unit Testing

### MessageQueue Module

The MessageQueue handles ZMQ communication between client and server.

**Key behaviors to test**:

| Behavior | Test Case |
|----------|-----------|
| Connection establishment | Client connects to server via ZMQ |
| Message serialization | Commands are properly serialized |
| K8s mode (UMRS_SINGLE_DATA_PORT) | Fixed port used when env var set |
| Tunnel mode | Localhost returned for data transfers |
| Timeout handling | Requests timeout after specified duration |
| Error handling | Graceful handling of connection failures |

**Test File**: `tests/test_message_queue.py`

```python
"""Unit tests for MessageQueue module.

Tests ZMQ-based client-server communication including:
- Connection handling
- Message serialization/deserialization
- K8s mode (UMRS_SINGLE_DATA_PORT)
- Tunnel mode (UMRS_TUNNEL_MODE)
- Error handling
"""
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock

class TestMessageQueueInit:
    """Tests for MessageQueue initialization."""

    def test_default_ports(self):
        """Default ports should be 5555 (control) and dynamic (data)."""
        with patch.dict(os.environ, {}, clear=True):
            # Import after environment is set
            from umrs.MessageQueue import MessageQueue
            mq = MessageQueue(host="localhost", port=5555)
            assert mq.control_port == 5555

    def test_custom_ports(self):
        """Custom ports should be respected."""
        with patch.dict(os.environ, {'UMRS_DATA_PORT': '6000'}):
            from umrs.MessageQueue import MessageQueue
            mq = MessageQueue(host="localhost", port=5555)
            # Verify custom data port is used


class TestK8sMode:
    """Tests for Kubernetes-compatible single data port mode."""

    def test_single_data_port_enabled(self):
        """UMRS_SINGLE_DATA_PORT=true should use fixed port."""
        with patch.dict(os.environ, {
            'UMRS_SINGLE_DATA_PORT': 'true',
            'UMRS_DATA_PORT': '5556'
        }):
            from umrs.MessageQueue import MessageQueue
            mq = MessageQueue(host="umrs-server", port=5555)
            
            # When single data port is enabled, hostname should be preserved
            # and fixed port should be used
            endpoint = mq.get_data_endpoint()
            assert '5556' in endpoint or mq.data_port == 5556

    def test_single_data_port_disabled(self):
        """Without UMRS_SINGLE_DATA_PORT, dynamic ports are used."""
        with patch.dict(os.environ, {'UMRS_SINGLE_DATA_PORT': 'false'}):
            from umrs.MessageQueue import MessageQueue
            mq = MessageQueue(host="localhost", port=5555)
            # Dynamic port allocation


class TestTunnelMode:
    """Tests for SSH tunnel compatibility mode."""

    def test_tunnel_mode_uses_localhost(self):
        """Tunnel mode should return localhost for all data transfers."""
        with patch.dict(os.environ, {
            'UMRS_TUNNEL_MODE': 'true',
            'UMRS_DATA_PORT': '5556'
        }):
            from umrs.MessageQueue import MessageQueue
            mq = MessageQueue(host="remote-host", port=5555)
            
            # In tunnel mode, data endpoint should use localhost
            # because SSH tunnel forwards to localhost


class TestMessageSerialization:
    """Tests for message serialization/deserialization."""

    def test_command_serialization(self):
        """Commands should be serialized to JSON."""
        command = {
            "action": "reload",
            "user": "testuser",
            "params": {"force": True}
        }
        serialized = json.dumps(command)
        deserialized = json.loads(serialized)
        assert deserialized == command

    def test_binary_data_handling(self):
        """Binary data should be handled separately from JSON."""
        # Binary data is sent via data socket, not control socket
        pass


class TestErrorHandling:
    """Tests for error handling in MessageQueue."""

    def test_connection_timeout(self):
        """Connection should timeout if server unavailable."""
        from umrs.MessageQueue import MessageQueue
        mq = MessageQueue(host="nonexistent", port=9999)
        
        with pytest.raises(Exception):  # Specify actual exception
            mq.connect(timeout=1)

    def test_invalid_response_handling(self):
        """Invalid server responses should be handled gracefully."""
        pass
```

### Client Module

**Key behaviors to test**:

| Behavior | Test Case |
|----------|-----------|
| Command sending | All commands properly formatted |
| Response parsing | Server responses correctly parsed |
| File operations | Upload/download work correctly |
| Error handling | Server errors properly propagated |

**Test File**: `tests/test_client.py`

```python
"""Unit tests for UMRS client library.

Tests client-side operations including:
- Command formatting
- Server communication
- File transfer
- Error handling
"""
import pytest
from unittest.mock import Mock, patch

class TestClientCommands:
    """Tests for client command operations."""

    @pytest.fixture
    def mock_mq(self):
        """Create a mock MessageQueue."""
        with patch('umrs.client_umrs.MessageQueue') as mock:
            yield mock.return_value

    def test_reload_command(self, mock_mq):
        """Reload command should be properly formatted."""
        from umrs.client_umrs import UMRSClient
        
        mock_mq.send_command.return_value = {"status": "success"}
        
        client = UMRSClient(host="localhost", port=5555)
        result = client.reload(user="testuser")
        
        mock_mq.send_command.assert_called_once()
        call_args = mock_mq.send_command.call_args
        assert call_args[0][0]["action"] == "reload"
        assert call_args[0][0]["user"] == "testuser"

    def test_check_command(self, mock_mq):
        """Check command should return server status."""
        from umrs.client_umrs import UMRSClient
        
        mock_mq.send_command.return_value = {
            "status": "success",
            "objects": {"images": 10, "texts": 5}
        }
        
        client = UMRSClient(host="localhost", port=5555)
        result = client.check(user="testuser")
        
        assert result["status"] == "success"

    def test_list_command(self, mock_mq):
        """List command should return file listing."""
        from umrs.client_umrs import UMRSClient
        
        mock_mq.send_command.return_value = {
            "status": "success",
            "files": ["file1.txt", "file2.jpg"]
        }
        
        client = UMRSClient(host="localhost", port=5555)
        result = client.list_files(user="testuser")
        
        assert len(result["files"]) == 2


class TestClientFileOperations:
    """Tests for file upload/download operations."""

    def test_file_upload_chunks(self):
        """Large files should be uploaded in chunks."""
        pass

    def test_file_download_reconstruction(self):
        """Downloaded chunks should be properly reconstructed."""
        pass


class TestClientErrorHandling:
    """Tests for client error handling."""

    def test_server_error_propagation(self, mock_mq):
        """Server errors should be propagated to caller."""
        from umrs.client_umrs import UMRSClient
        
        mock_mq.send_command.return_value = {
            "status": "error",
            "message": "User not found"
        }
        
        client = UMRSClient(host="localhost", port=5555)
        result = client.check(user="nonexistent")
        
        assert result["status"] == "error"

    def test_network_error_handling(self, mock_mq):
        """Network errors should be handled gracefully."""
        from umrs.client_umrs import UMRSClient
        
        mock_mq.send_command.side_effect = ConnectionError("Connection refused")
        
        client = UMRSClient(host="localhost", port=5555)
        
        with pytest.raises(ConnectionError):
            client.check(user="testuser")
```

### Embedding Worker Module

**Key behaviors to test**:

| Behavior | Test Case |
|----------|-----------|
| Model loading | Models load correctly on specified device |
| Text embedding | Text produces correct dimension vectors |
| Image embedding | Images produce correct dimension vectors |
| Batch processing | Batches processed efficiently |
| Error handling | Corrupt files handled gracefully |

**Test File**: `tests/test_embedding_worker.py`

```python
"""Unit tests for embedding worker.

Tests embedding generation including:
- Model loading
- Text embedding
- Image embedding
- Batch processing
- Error handling
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestModelLoading:
    """Tests for embedding model loading."""

    @patch('umrs.embedding_worker.SentenceTransformer')
    def test_text_model_loading(self, mock_st):
        """Text embedding model should load correctly."""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        from umrs.embedding_worker import EmbeddingWorker
        worker = EmbeddingWorker(model="intfloat/e5-base-v2")
        
        mock_st.assert_called_once()

    @patch('umrs.embedding_worker.CLIPModel')
    def test_clip_model_loading(self, mock_clip):
        """CLIP model should load for image embeddings."""
        mock_model = Mock()
        mock_clip.from_pretrained.return_value = mock_model
        
        from umrs.embedding_worker import EmbeddingWorker
        worker = EmbeddingWorker(model="clip-ViT-B-32", modality="image")
        
        mock_clip.from_pretrained.assert_called_once()

    def test_device_selection_cuda(self):
        """CUDA should be used when available and specified."""
        with patch('torch.cuda.is_available', return_value=True):
            from umrs.embedding_worker import EmbeddingWorker
            worker = EmbeddingWorker(device="cuda")
            assert "cuda" in str(worker.device)

    def test_device_fallback_cpu(self):
        """Should fall back to CPU when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            from umrs.embedding_worker import EmbeddingWorker
            worker = EmbeddingWorker(device="cuda")
            # Should not fail, should fall back to CPU


class TestTextEmbedding:
    """Tests for text embedding generation."""

    @pytest.fixture
    def mock_worker(self):
        """Create worker with mocked model."""
        with patch('umrs.embedding_worker.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(768)
            mock_st.return_value = mock_model
            
            from umrs.embedding_worker import EmbeddingWorker
            worker = EmbeddingWorker(model="intfloat/e5-base-v2")
            yield worker, mock_model

    def test_text_embedding_dimensions(self, mock_worker):
        """Text embeddings should have correct dimensions."""
        worker, mock_model = mock_worker
        
        embedding = worker.embed_text("Hello world")
        
        assert len(embedding) == 768

    def test_empty_text_handling(self, mock_worker):
        """Empty text should be handled gracefully."""
        worker, mock_model = mock_worker
        mock_model.encode.return_value = np.zeros(768)
        
        embedding = worker.embed_text("")
        
        assert len(embedding) == 768


class TestImageEmbedding:
    """Tests for image embedding generation."""

    @pytest.fixture
    def mock_clip_worker(self):
        """Create worker with mocked CLIP model."""
        with patch('umrs.embedding_worker.CLIPModel') as mock_clip:
            with patch('umrs.embedding_worker.CLIPProcessor') as mock_proc:
                mock_model = Mock()
                mock_model.get_image_features.return_value = Mock(
                    cpu=Mock(return_value=Mock(
                        numpy=Mock(return_value=np.random.rand(1, 512))
                    ))
                )
                mock_clip.from_pretrained.return_value = mock_model
                
                from umrs.embedding_worker import EmbeddingWorker
                worker = EmbeddingWorker(model="clip-ViT-B-32", modality="image")
                yield worker

    def test_image_embedding_dimensions(self, mock_clip_worker):
        """Image embeddings should have correct dimensions (512 for CLIP)."""
        worker = mock_clip_worker
        
        # Mock image loading
        with patch('PIL.Image.open'):
            embedding = worker.embed_image("test.jpg")
        
        assert len(embedding) == 512

    def test_corrupt_image_handling(self, mock_clip_worker):
        """Corrupt images should raise appropriate error."""
        worker = mock_clip_worker
        
        with patch('PIL.Image.open', side_effect=IOError("Cannot identify image")):
            with pytest.raises(IOError):
                worker.embed_image("corrupt.jpg")


class TestBatchProcessing:
    """Tests for batch embedding processing."""

    def test_batch_text_embedding(self):
        """Batch processing should be more efficient."""
        with patch('umrs.embedding_worker.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(5, 768)
            mock_st.return_value = mock_model
            
            from umrs.embedding_worker import EmbeddingWorker
            worker = EmbeddingWorker(model="intfloat/e5-base-v2")
            
            texts = ["text1", "text2", "text3", "text4", "text5"]
            embeddings = worker.embed_batch(texts)
            
            # Should call encode once with batch, not 5 times
            assert mock_model.encode.call_count == 1
            assert len(embeddings) == 5
```

---

## Integration Testing

### Server Integration Tests

**Test File**: `tests/test_server.py`

```python
"""Integration tests for UMRS server.

Tests server operations with real (or containerized) dependencies:
- MinIO (S3)
- PostgreSQL
- ZMQ communication
"""
import pytest
import subprocess
import time
import os

@pytest.fixture(scope="module")
def docker_services():
    """Start Docker services for testing."""
    subprocess.run(
        ["docker-compose", "-f", "docker/docker-compose.yml", "up", "-d"],
        cwd=os.path.dirname(__file__) + "/..",
        check=True
    )
    time.sleep(10)  # Wait for services to be ready
    yield
    subprocess.run(
        ["docker-compose", "-f", "docker/docker-compose.yml", "down", "-v"],
        cwd=os.path.dirname(__file__) + "/..",
        check=True
    )


class TestServerOperations:
    """Integration tests for server operations."""

    def test_server_starts_and_accepts_connections(self, docker_services):
        """Server should start and accept client connections."""
        from umrs.client_umrs import UMRSClient
        
        client = UMRSClient(host="localhost", port=5555)
        result = client.check(user="testuser")
        
        assert result["status"] == "success"

    def test_reload_processes_s3_objects(self, docker_services):
        """Reload should scan S3 and update database."""
        from umrs.client_umrs import UMRSClient
        
        # Upload test file to S3 first
        # ...
        
        client = UMRSClient(host="localhost", port=5555)
        result = client.reload(user="testuser")
        
        assert result["status"] == "success"
        
        # Verify database was updated
        result = client.check(user="testuser")
        assert result["objects"]["total"] > 0


class TestS3Integration:
    """Tests for S3/MinIO integration."""

    def test_file_upload_creates_s3_object(self, docker_services):
        """Uploading a file should create S3 object."""
        from umrs.client_umrs import UMRSClient
        
        client = UMRSClient(host="localhost", port=5555)
        
        # Create test file
        with open("/tmp/test_upload.txt", "w") as f:
            f.write("test content")
        
        result = client.upload(
            user="testuser",
            filepath="/tmp/test_upload.txt",
            remote_path="test_upload.txt"
        )
        
        assert result["status"] == "success"

    def test_file_download_retrieves_content(self, docker_services):
        """Downloading should retrieve file content."""
        pass


class TestDatabaseIntegration:
    """Tests for PostgreSQL/pgvector integration."""

    def test_embedding_stored_in_pgvector(self, docker_services):
        """Generated embeddings should be stored in pgvector."""
        pass

    def test_similarity_search_returns_results(self, docker_services):
        """Similarity search should return relevant results."""
        pass
```

---

## API Testing

### Shell-Based API Tests

The `test_umrs_api.sh` script provides quick API validation:

```bash
#!/bin/bash
# test_umrs_api.sh - Test UMRS API operations

set -e

UMRS_HOST="${UMRS_HOST:-localhost}"
UMRS_PORT="${UMRS_PORT:-5555}"

echo "Testing UMRS API at ${UMRS_HOST}:${UMRS_PORT}"

# Test 1: Server connectivity
echo "=== Test 1: Server Connectivity ==="
python3 -c "
from umrs.client_umrs import UMRSClient
client = UMRSClient(host='${UMRS_HOST}', port=${UMRS_PORT})
result = client.check(user='testuser')
assert result['status'] == 'success', f'Check failed: {result}'
print('✓ Server connectivity OK')
"

# Test 2: Reload operation
echo "=== Test 2: Reload Operation ==="
python3 -c "
from umrs.client_umrs import UMRSClient
client = UMRSClient(host='${UMRS_HOST}', port=${UMRS_PORT})
result = client.reload(user='testuser')
assert result['status'] == 'success', f'Reload failed: {result}'
print('✓ Reload operation OK')
"

# Test 3: File upload
echo "=== Test 3: File Upload ==="
echo "test content" > /tmp/test_file.txt
python3 -c "
from umrs.client_umrs import UMRSClient
client = UMRSClient(host='${UMRS_HOST}', port=${UMRS_PORT})
result = client.upload(user='testuser', filepath='/tmp/test_file.txt')
assert result['status'] == 'success', f'Upload failed: {result}'
print('✓ File upload OK')
"

echo "=== All API tests passed ==="
```

---

## Database Testing

### pgvector Tests

```python
"""Tests for pgvector operations."""
import pytest
import psycopg2
import numpy as np

@pytest.fixture
def db_connection():
    """Create database connection for testing."""
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
        database=os.environ.get("PGDATABASE", "umrs")
    )
    yield conn
    conn.close()


class TestPgvectorOperations:
    """Tests for pgvector embedding operations."""

    def test_embedding_insertion(self, db_connection):
        """Embeddings should be insertable."""
        cursor = db_connection.cursor()
        
        embedding = np.random.rand(768).tolist()
        cursor.execute(
            "INSERT INTO embeddings (file_id, embedding) VALUES (%s, %s)",
            (1, embedding)
        )
        db_connection.commit()

    def test_similarity_search(self, db_connection):
        """Similarity search should return ordered results."""
        cursor = db_connection.cursor()
        
        query_embedding = np.random.rand(768).tolist()
        cursor.execute("""
            SELECT file_id, embedding <-> %s::vector AS distance
            FROM embeddings
            ORDER BY distance
            LIMIT 5
        """, (query_embedding,))
        
        results = cursor.fetchall()
        # Results should be ordered by distance
        distances = [r[1] for r in results]
        assert distances == sorted(distances)
```

---

## Running Tests

### Quick Commands

```bash
# All unit tests
cd umrs
pytest tests/ -v

# Specific test file
pytest tests/test_message_queue.py -v

# With coverage
pytest tests/ --cov=src/umrs --cov-report=html

# Integration tests (requires Docker)
cd umrs/docker
./test_umrs_api.sh

# Full E2E (requires Docker)
./test_rag_end_to_end.sh
```

### Test Markers

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "integration: Integration tests (needs Docker)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full stack)")
    config.addinivalue_line("markers", "slow: Slow tests")
```

Running by marker:
```bash
pytest tests/ -m unit -v       # Only unit tests
pytest tests/ -m integration   # Only integration tests
pytest tests/ -m "not slow"    # Skip slow tests
```

---

## Test Fixtures

### conftest.py

```python
"""Shared test fixtures for UMRS tests."""
import pytest
import os
from unittest.mock import Mock, patch

@pytest.fixture
def mock_s3_client():
    """Mock boto3 S3 client."""
    with patch('boto3.client') as mock:
        client = Mock()
        mock.return_value = client
        yield client

@pytest.fixture
def mock_pg_connection():
    """Mock PostgreSQL connection."""
    with patch('psycopg2.connect') as mock:
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        mock.return_value = conn
        yield conn, cursor

@pytest.fixture
def test_env():
    """Set up test environment variables."""
    env_vars = {
        'UMRS_HOST': 'localhost',
        'UMRS_PORT': '5555',
        'S3_ENDPOINT': 'http://localhost:9000',
        'PGHOST': 'localhost',
        'PGPORT': '5432'
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is sample text content for testing.")
    return file_path

@pytest.fixture
def sample_image_file(tmp_path):
    """Create a sample image file for testing."""
    from PIL import Image
    file_path = tmp_path / "sample.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(file_path)
    return file_path
```

---

## Mocking Guidelines

### When to Mock

| Scenario | Mock? | Reason |
|----------|-------|--------|
| External services (S3, DB) | Yes (unit) | Isolation |
| File system operations | Sometimes | Speed |
| Network calls | Yes | Reliability |
| Time-dependent code | Yes | Reproducibility |

### Mocking Examples

```python
# Mock S3 operations
@patch('boto3.client')
def test_s3_upload(mock_client):
    mock_s3 = Mock()
    mock_client.return_value = mock_s3
    
    # Test code that uses S3
    
    mock_s3.put_object.assert_called_once()

# Mock database
@patch('psycopg2.connect')
def test_db_query(mock_connect):
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.fetchall.return_value = [("row1",), ("row2",)]
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    
    # Test code that queries database

# Mock embedding model
@patch('sentence_transformers.SentenceTransformer')
def test_embedding(mock_st):
    mock_model = Mock()
    mock_model.encode.return_value = np.zeros(768)
    mock_st.return_value = mock_model
    
    # Test embedding generation
```

---

## TDD Examples

### Example: Adding a New Command

**Requirement**: Add a `delete` command to remove files from S3.

**Step 1**: Write failing test

```python
# tests/test_client.py

def test_delete_command_removes_file():
    """Delete command should remove file from S3."""
    with patch('umrs.client_umrs.MessageQueue') as mock_mq:
        mock_mq.return_value.send_command.return_value = {"status": "success"}
        
        from umrs.client_umrs import UMRSClient
        client = UMRSClient(host="localhost", port=5555)
        
        result = client.delete(user="testuser", filepath="test.txt")
        
        assert result["status"] == "success"
        mock_mq.return_value.send_command.assert_called_with({
            "action": "delete",
            "user": "testuser",
            "filepath": "test.txt"
        })
```

**Step 2**: Run test (fails)

```bash
pytest tests/test_client.py::test_delete_command_removes_file -v
# FAILED - AttributeError: 'UMRSClient' object has no attribute 'delete'
```

**Step 3**: Implement minimum code

```python
# src/umrs/client_umrs.py

class UMRSClient:
    # ... existing code ...
    
    def delete(self, user: str, filepath: str) -> dict:
        """Delete a file from S3.
        
        Args:
            user: Username
            filepath: Path to file in S3
            
        Returns:
            dict: Server response
        """
        return self.mq.send_command({
            "action": "delete",
            "user": user,
            "filepath": filepath
        })
```

**Step 4**: Run test (passes)

```bash
pytest tests/test_client.py::test_delete_command_removes_file -v
# PASSED
```

**Step 5**: Add edge cases

```python
def test_delete_nonexistent_file_returns_error():
    """Deleting non-existent file should return error."""
    with patch('umrs.client_umrs.MessageQueue') as mock_mq:
        mock_mq.return_value.send_command.return_value = {
            "status": "error",
            "message": "File not found"
        }
        
        from umrs.client_umrs import UMRSClient
        client = UMRSClient(host="localhost", port=5555)
        
        result = client.delete(user="testuser", filepath="nonexistent.txt")
        
        assert result["status"] == "error"

def test_delete_validates_filepath():
    """Delete should validate filepath format."""
    from umrs.client_umrs import UMRSClient
    client = UMRSClient(host="localhost", port=5555)
    
    with pytest.raises(ValueError):
        client.delete(user="testuser", filepath="")  # Empty path
```

---

## Appendix: Test Requirements

### requirements.txt

```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0
pytest-timeout>=2.1.0
httpx>=0.24.0
fakeredis>=2.10.0
moto>=4.0.0
psycopg2-binary>=2.9.0
numpy>=1.24.0
Pillow>=9.0.0
```

### Installing Test Dependencies

```bash
cd umrs
pip install -r tests/requirements.txt
```
