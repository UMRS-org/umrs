# UMRS Demos

End-to-end demonstrations of UMRS capabilities for the AI-for-Science Platform.

## Available Demos

### e2e_rivault_slurm_rag.py

Complete workflow demonstration showing:
1. **Model Registry**: Register and query models
2. **Orchestration**: Submit Mini-RiVault jobs via Slurm
3. **RAG Pipeline**: Hybrid search (BM25 + vector similarity)
4. **LLM Completion**: Query the provisioned endpoint with RAG context

#### Quick Start

```bash
# Mock mode (no Slurm required)
cd umrs
python demos/e2e_rivault_slurm_rag.py --mock

# Custom query
python demos/e2e_rivault_slurm_rag.py --mock \
    --query "How does HPC help scientific computing?"

# Live mode (on HPC cluster with Slurm)
python demos/e2e_rivault_slurm_rag.py --live \
    --model-path /models/qwen-0.5b \
    --partition gpu
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mock` | Use mock implementations (no Slurm) | Yes |
| `--live` | Use actual Slurm orchestrator | No |
| `--model-path` | Path to model weights | `/models/rivault-gpt-oss` |
| `--partition` | Slurm partition name | `gpu` |
| `--query` | Question to ask the RAG system | (seismic wave example) |

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `RIVAULT_ORCHESTRATOR` | Backend: slurm, k8s, compose |
| `SLURM_PARTITION` | Default partition |
| `LITELLM_PORT` | LiteLLM proxy port |
| `LITELLM_MASTER_KEY` | API key for LiteLLM |
| `DEMO_MODEL_PATH` | Default model path |

#### Sample Output

```
╔════════════════════════════════════════════════════════════════════╗
║             E2E DEMO: RiVault + Slurm + RAG                        ║
║                        Mode: MOCK                                   ║
╚════════════════════════════════════════════════════════════════════╝

======================================================================
STEP 1: Model Registry
======================================================================
✓ Model entry created:
  - ID: demo-qwen-0.5b
  - Name: Qwen-0.5B Demo
  - Scope: project
  - Path: /models/rivault-gpt-oss

======================================================================
STEP 2: Orchestration (Slurm → Mini-RiVault)
======================================================================
✓ Using MOCK orchestrator (no Slurm)
✓ Mock endpoint ready:
  - Host: localhost
  - Port: 11434

======================================================================
STEP 3: RAG Pipeline (Hybrid Search)
======================================================================
Query: "How does seismic wave propagation work..."

→ Running hybrid search (BM25 + Vector)...
✓ Search results (top 3):
  [1] Seismic Wave Propagation
      Score: 0.0142 (BM25: 5.2341, Vector: 0.6821)
  [2] Machine Learning for Geophysics
      Score: 0.0138 (BM25: 3.1245, Vector: 0.7123)
  ...

======================================================================
STEP 4: LLM Completion (via RiVault)
======================================================================
✓ [MOCK] Response generated

──────────────────────────────────────────────────
Answer:
Based on the retrieved documents, seismic wave propagation follows
the elastic wave equation. P-waves travel faster than S-waves...
──────────────────────────────────────────────────

======================================================================
✓ Demo completed successfully!
======================================================================
```

## Adding New Demos

When adding demos, follow this pattern:

1. Create a Python script in `demos/`
2. Support both `--mock` and `--live` modes
3. Use `asyncio` for async operations
4. Include comprehensive docstrings
5. Update this README

## Related Documentation

- [UMRS Documentation](../doc/)
- [RAG Pipeline Tests](../tests/test_rag_pipeline.py)
- [Orchestrator Module](../src/umrs/orchestrator/)
- [Model Registry](../src/umrs/model_registry/)
