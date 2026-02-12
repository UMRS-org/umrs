#!/usr/bin/env python3
"""
End-to-End Demo: RiVault + Slurm + RAG

Demonstrates the full UMRS workflow for AI-for-Science Platform:
1. Model Registry: List/register models
2. Orchestration: Submit Mini-RiVault job via Slurm
3. RAG Pipeline: Hybrid search with BM25 + vector similarity
4. LLM Completion: Query via the provisioned RiVault endpoint

This demo can run in two modes:
- LIVE: Actually submits Slurm jobs (requires HPC cluster)
- MOCK: Simulates the flow for testing/development

Usage:
    # Live mode (on HPC cluster)
    python demos/e2e_rivault_slurm_rag.py --live --model-path /models/qwen-0.5b

    # Mock mode (for development)
    python demos/e2e_rivault_slurm_rag.py --mock

    # Show available options
    python demos/e2e_rivault_slurm_rag.py --help

Environment Variables:
    RIVAULT_ORCHESTRATOR   - Orchestrator backend (slurm/k8s/compose)
    SLURM_PARTITION        - Slurm partition name
    LITELLM_PORT           - LiteLLM proxy port
    LITELLM_MASTER_KEY     - API key for LiteLLM
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Add UMRS source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_demo")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DemoConfig:
    """Configuration for E2E demo."""

    # Mode
    live_mode: bool = False  # If False, use mocks

    # Model configuration
    model_path: str = field(
        default_factory=lambda: os.environ.get(
            "DEMO_MODEL_PATH", "/models/rivault-gpt-oss"
        )
    )
    model_name: str = "rivault-gpt-oss-static"

    # Slurm configuration
    partition: str = field(
        default_factory=lambda: os.environ.get("SLURM_PARTITION", "gpu")
    )
    time_limit: str = "00:30:00"
    gpus: int = 1

    # LiteLLM configuration
    litellm_port: int = field(
        default_factory=lambda: int(os.environ.get("LITELLM_PORT", "11434"))
    )
    litellm_key: str = field(
        default_factory=lambda: os.environ.get("LITELLM_MASTER_KEY", "sk-ll-demo")
    )

    # RAG configuration
    embedding_dim: int = 384
    top_k: int = 3

    # Timeouts
    ready_timeout: int = 300


# =============================================================================
# Sample Documents for RAG Demo
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Seismic Wave Propagation",
        "content": """
        Seismic wave propagation in Earth's crust follows the elastic wave equation.
        P-waves (primary waves) are compressional waves that travel faster than S-waves
        (secondary waves), which are shear waves. The velocity depends on the elastic
        properties and density of the medium. In complex geological structures, waves
        undergo reflection, refraction, and diffraction at interfaces.
        """.strip(),
    },
    {
        "id": "doc2",
        "title": "Machine Learning for Geophysics",
        "content": """
        Deep learning has revolutionized seismic interpretation. Convolutional neural
        networks (CNNs) are used for fault detection and horizon picking. Recurrent
        networks help with sequential seismic data analysis. Physics-informed neural
        networks (PINNs) combine data-driven learning with wave equation constraints
        for more accurate subsurface imaging.
        """.strip(),
    },
    {
        "id": "doc3",
        "title": "HPC for Scientific Computing",
        "content": """
        High-performance computing enables large-scale scientific simulations.
        GPU acceleration with CUDA/ROCm provides massive parallelism for numerical
        methods. Slurm workload manager schedules jobs across compute nodes.
        Containerization with Singularity ensures reproducibility in HPC environments.
        MPI and collective communication optimize distributed computations.
        """.strip(),
    },
    {
        "id": "doc4",
        "title": "RiVault LLM Platform",
        "content": """
        RiVault is an AI-for-Science platform designed for HPC environments.
        It provides on-demand LLM inference with vLLM backend, proxied through
        LiteLLM for OpenAI API compatibility. Models are pre-staged on the
        filesystem and jobs are scheduled via Slurm. The system supports RAG
        for domain-specific question answering.
        """.strip(),
    },
    {
        "id": "doc5",
        "title": "Numerical Methods in Geodynamics",
        "content": """
        Finite element methods (FEM) and spectral element methods (SEM) are
        fundamental for solving partial differential equations in geodynamics.
        Adaptive mesh refinement improves accuracy in regions with high gradients.
        Time-stepping schemes must balance accuracy with stability constraints.
        Preconditioning is essential for efficient iterative solvers.
        """.strip(),
    },
]


# =============================================================================
# Mock Implementations (for development/testing)
# =============================================================================


class MockOrchestrator:
    """Mock orchestrator for demo without actual Slurm."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self._job_counter = 0

    async def submit_job(self, **kwargs) -> dict:
        """Simulate job submission."""
        self._job_counter += 1
        job_id = f"mock-{self._job_counter}"
        logger.info(f"[MOCK] Submitted job: {job_id}")
        return {
            "job_id": job_id,
            "status": "PENDING",
            "model_path": kwargs.get("model_path", self.config.model_path),
        }

    async def wait_for_ready(self, job_id: str) -> dict:
        """Simulate waiting for job readiness."""
        logger.info(f"[MOCK] Waiting for job {job_id}...")
        await asyncio.sleep(1)  # Brief simulation
        return {
            "job_id": job_id,
            "status": "READY",
            "host": "localhost",
            "port": self.config.litellm_port,
        }

    async def cleanup(self, job_id: str) -> None:
        """Simulate job cleanup."""
        logger.info(f"[MOCK] Cleaned up job: {job_id}")


class MockEmbeddingModel:
    """Mock embedding model for demo."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic mock embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Use hash to get deterministic but unique embeddings
            np.random.seed(hash(text) % (2**32))
            vec = np.random.randn(self.dim)
            vec = vec / np.linalg.norm(vec)  # Normalize
            embeddings.append(vec)
        return np.array(embeddings)


class MockLLMClient:
    """Mock LLM client for demo."""

    async def chat_completion(
        self,
        messages: list[dict],
        model: str = "rivault-gpt-oss-static",
        **kwargs,
    ) -> dict:
        """Simulate chat completion."""
        user_msg = messages[-1].get("content", "") if messages else ""

        # Generate a mock response based on the query (check more specific first)
        if "hpc" in user_msg.lower() or "slurm" in user_msg.lower() or "scientific computing" in user_msg.lower():
            response = (
                "HPC environments like RCCS use Slurm for job scheduling. GPU acceleration "
                "with CUDA enables parallel computation. Singularity containers ensure "
                "reproducibility. RiVault leverages this infrastructure for on-demand "
                "LLM inference with vLLM backend."
            )
        elif "seismic" in user_msg.lower():
            response = (
                "Based on the retrieved documents, seismic wave propagation follows "
                "the elastic wave equation. P-waves travel faster than S-waves due to "
                "their compressional nature. Modern deep learning approaches using CNNs "
                "and physics-informed neural networks have improved seismic interpretation."
            )
        else:
            response = (
                "I can help answer questions about scientific computing, geophysics, "
                "machine learning, and the RiVault AI platform. The retrieved context "
                "provides domain-specific knowledge for accurate responses."
            )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": model,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }


# =============================================================================
# Demo Steps
# =============================================================================


async def demo_model_registry(config: DemoConfig) -> None:
    """Demonstrate model registry operations."""
    print("\n" + "=" * 70)
    print("STEP 1: Model Registry")
    print("=" * 70)

    try:
        from umrs.model_registry import ModelEntry, ModelScope, ModelStatus, ModelType

        # Create a demo model entry
        model = ModelEntry(
            model_id="demo-qwen-0.5b",
            name="Qwen-0.5B Demo",
            scope=ModelScope.PROJECT,
            owner_id="demo-user",
            project_id="ai-for-science",
            model_type=ModelType.LLM,
            status=ModelStatus.AVAILABLE,
            storage_ref=config.model_path,
            description="Small model for demo purposes",
        )

        print(f"✓ Model entry created:")
        print(f"  - ID: {model.model_id}")
        print(f"  - Name: {model.name}")
        print(f"  - Scope: {model.scope.value}")
        print(f"  - Path: {model.storage_ref}")
        print(f"  - Status: {model.status.value}")

    except Exception as e:
        print(f"⚠ Model registry warning: {e}")
        print("  (Showing mock model info)")
        print(f"  - ID: demo-qwen-0.5b")
        print(f"  - Name: Qwen-0.5B Demo")
        print(f"  - Path: {config.model_path}")


async def demo_orchestration(config: DemoConfig) -> dict:
    """Demonstrate Mini-RiVault orchestration via Slurm."""
    print("\n" + "=" * 70)
    print("STEP 2: Orchestration (Slurm → Mini-RiVault)")
    print("=" * 70)

    if config.live_mode:
        # Use real orchestrator
        try:
            from umrs.orchestrator import get_orchestrator, OrchestratorConfig

            orch_config = OrchestratorConfig(
                model_path=config.model_path,
                slurm_partition=config.partition,
                slurm_time=config.time_limit,
                slurm_gpus=config.gpus,
                litellm_port=config.litellm_port,
                litellm_master_key=config.litellm_key,
            )
            orchestrator = get_orchestrator(backend="slurm", config=orch_config)

            print("✓ Using LIVE Slurm orchestrator")
            print(f"  - Partition: {config.partition}")
            print(f"  - Time limit: {config.time_limit}")
            print(f"  - GPUs: {config.gpus}")

            # Submit job
            print("\n→ Submitting Mini-RiVault job...")
            job_info = await orchestrator.submit_job(model_path=config.model_path)
            job_id = job_info.job_id
            print(f"  Job submitted: {job_id}")

            # Wait for ready
            print(f"\n→ Waiting for job to be ready (timeout: {config.ready_timeout}s)...")
            endpoint = await orchestrator.wait_for_ready(job_id, timeout=config.ready_timeout)

            return {
                "job_id": job_id,
                "host": endpoint.host,
                "port": endpoint.port,
                "orchestrator": orchestrator,
            }

        except Exception as e:
            print(f"✗ Live orchestration failed: {e}")
            print("  Falling back to mock mode...")
            config.live_mode = False

    # Mock mode
    mock_orch = MockOrchestrator(config)
    print("✓ Using MOCK orchestrator (no Slurm)")

    job_info = await mock_orch.submit_job(model_path=config.model_path)
    endpoint = await mock_orch.wait_for_ready(job_info["job_id"])

    print(f"\n✓ Mock endpoint ready:")
    print(f"  - Host: {endpoint['host']}")
    print(f"  - Port: {endpoint['port']}")

    return {
        "job_id": endpoint["job_id"],
        "host": endpoint["host"],
        "port": endpoint["port"],
        "orchestrator": mock_orch,
    }


async def demo_rag_pipeline(config: DemoConfig, query: str) -> dict:
    """Demonstrate RAG pipeline with hybrid search."""
    print("\n" + "=" * 70)
    print("STEP 3: RAG Pipeline (Hybrid Search)")
    print("=" * 70)

    print(f"Query: \"{query}\"")

    # Prepare documents with embeddings
    embedding_model = MockEmbeddingModel(dim=config.embedding_dim)

    print("\n→ Generating embeddings for documents...")
    doc_texts = [d["content"] for d in SAMPLE_DOCUMENTS]
    doc_embeddings = embedding_model.encode(doc_texts)

    documents_with_embeddings = [
        (doc["id"], doc["content"], emb.tolist())
        for doc, emb in zip(SAMPLE_DOCUMENTS, doc_embeddings)
    ]

    print(f"  Processed {len(SAMPLE_DOCUMENTS)} documents")

    # Generate query embedding
    query_embedding = embedding_model.encode([query])[0].tolist()

    # Use RAG pipeline
    try:
        from umrs.rag_pipeline import (
            HybridSearcher,
            ContextAssembler,
            RAGContext,
        )

        print("\n→ Running hybrid search (BM25 + Vector)...")
        searcher = HybridSearcher(vector_weight=0.6, bm25_weight=0.4)

        results = await searcher.search(
            query=query,
            query_embedding=query_embedding,
            documents=documents_with_embeddings,
            top_k=config.top_k,
        )

        print(f"\n✓ Search results (top {config.top_k}):")
        for i, r in enumerate(results, 1):
            doc = next((d for d in SAMPLE_DOCUMENTS if d["id"] == r.document_id), None)
            title = doc["title"] if doc else "Unknown"
            print(f"  [{i}] {title}")
            print(f"      Score: {r.score:.4f} (BM25: {r.bm25_score:.4f}, Vector: {r.vector_score:.4f})")

        # Assemble context
        print("\n→ Assembling context...")
        assembler = ContextAssembler(
            max_tokens=2000,
            enable_injection_scanning=True,
        )

        context = await assembler.assemble(query=query, results=results)

        print(f"✓ Context assembled:")
        print(f"  - Chunks: {context.total_chunks}")
        print(f"  - Est. tokens: {context.total_tokens_estimate}")
        print(f"  - Sources: {[s['source'] for s in context.sources]}")

        return {
            "results": results,
            "context": context,
            "query_embedding": query_embedding,
        }

    except ImportError as e:
        print(f"⚠ RAG pipeline import warning: {e}")
        # Return mock context
        return {
            "results": [],
            "context": {"text": "Mock context", "sources": []},
            "query_embedding": query_embedding,
        }


async def demo_llm_completion(
    config: DemoConfig,
    endpoint_info: dict,
    rag_context: dict,
    query: str,
) -> str:
    """Demonstrate LLM completion with RAG context."""
    print("\n" + "=" * 70)
    print("STEP 4: LLM Completion (via RiVault)")
    print("=" * 70)

    # Build prompt with RAG context
    context = rag_context.get("context")
    if hasattr(context, "context_text"):
        context_text = context.context_text
    else:
        context_text = str(context)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a scientific AI assistant. Answer questions based on "
                "the provided context. Cite sources using [n] notation."
            ),
        },
        {
            "role": "user",
            "content": f"{context_text}\n\nQuestion: {query}",
        },
    ]

    print(f"→ Sending request to LLM endpoint...")
    print(f"  - Host: {endpoint_info['host']}")
    print(f"  - Port: {endpoint_info['port']}")
    print(f"  - Model: {config.model_name}")

    if config.live_mode:
        # Use real HTTP client
        try:
            import httpx

            url = f"http://{endpoint_info['host']}:{endpoint_info['port']}/v1/chat/completions"
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    json={
                        "model": config.model_name,
                        "messages": messages,
                        "max_tokens": 512,
                        "temperature": 0.7,
                    },
                    headers={"Authorization": f"Bearer {config.litellm_key}"},
                )
                response.raise_for_status()
                result = response.json()

            answer = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            print(f"\n✓ Response received:")
            print(f"  - Tokens: {usage.get('total_tokens', 'N/A')}")

        except Exception as e:
            print(f"⚠ Live request failed: {e}")
            print("  Falling back to mock response...")
            mock_client = MockLLMClient()
            result = await mock_client.chat_completion(messages, config.model_name)
            answer = result["choices"][0]["message"]["content"]
    else:
        # Use mock client
        mock_client = MockLLMClient()
        result = await mock_client.chat_completion(messages, config.model_name)
        answer = result["choices"][0]["message"]["content"]
        print("\n✓ [MOCK] Response generated")

    print(f"\n{'─' * 50}")
    print(f"Answer:\n{answer}")
    print(f"{'─' * 50}")

    return answer


async def demo_cleanup(config: DemoConfig, endpoint_info: dict) -> None:
    """Cleanup resources."""
    print("\n" + "=" * 70)
    print("STEP 5: Cleanup")
    print("=" * 70)

    orchestrator = endpoint_info.get("orchestrator")
    job_id = endpoint_info.get("job_id")

    if orchestrator and job_id:
        if hasattr(orchestrator, "cleanup"):
            await orchestrator.cleanup(job_id)
            print(f"✓ Job {job_id} cleaned up")
        else:
            print(f"⚠ No cleanup method available for job {job_id}")
    else:
        print("✓ No resources to clean up")


# =============================================================================
# Main Demo Runner
# =============================================================================


async def run_demo(config: DemoConfig, query: str) -> None:
    """Run the complete E2E demo."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " E2E DEMO: RiVault + Slurm + RAG ".center(68) + "║")
    print("║" + f" Mode: {'LIVE' if config.live_mode else 'MOCK'} ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    endpoint_info = None
    try:
        # Step 1: Model Registry
        await demo_model_registry(config)

        # Step 2: Orchestration
        endpoint_info = await demo_orchestration(config)

        # Step 3: RAG Pipeline
        rag_context = await demo_rag_pipeline(config, query)

        # Step 4: LLM Completion
        await demo_llm_completion(config, endpoint_info, rag_context, query)

    finally:
        # Step 5: Cleanup
        if endpoint_info:
            await demo_cleanup(config, endpoint_info)

    print("\n" + "=" * 70)
    print("✓ Demo completed successfully!")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="E2E Demo: RiVault + Slurm + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Use live Slurm orchestrator (requires HPC cluster)",
    )
    mode_group.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock implementations (default)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("DEMO_MODEL_PATH", "/models/rivault-gpt-oss"),
        help="Path to model weights (default: $DEMO_MODEL_PATH)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=os.environ.get("SLURM_PARTITION", "gpu"),
        help="Slurm partition (default: gpu)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="How does seismic wave propagation work and how can machine learning improve seismic interpretation?",
        help="Question to ask the RAG system",
    )

    args = parser.parse_args()

    config = DemoConfig(
        live_mode=args.live,
        model_path=args.model_path,
        partition=args.partition,
    )

    asyncio.run(run_demo(config, args.query))


if __name__ == "__main__":
    main()
