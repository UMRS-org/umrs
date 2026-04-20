# Demo 18: Model Lifecycle

Demonstrates the full model lifecycle: **list → download → register → use → verify**.
Shows both system-scoped (shared) and user-scoped (private) model management.

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556, 9000)
- `umrs-client` installed and in PATH
- `huggingface_hub` installed (for `model download` only)

## Quick Start

```bash
cd demos/18_model_lifecycle
bash run_demo.sh
```

## What This Demo Does

1. **Pre-flight** — verifies server connectivity
2. **List existing models** — shows current system and user model inventories
3. **Verify or stage an embedding model** — reuse a visible shared model, or
   stage a user-scoped copy if your site exposes a writable model root
4. **Register a user model** — optionally uploads a local model directory as a
   user-scoped model (requires `DEMO_LOCAL_MODEL_UPLOAD=1`)
5. **Verify registration** — lists user models to confirm the new one appears
6. **Use model for inference** — uploads a text file to trigger embedding with
   the system model
7. **Final inventory** — shows updated system and user model lists

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. List all currently registered models
#    Shows models grouped by scope (system = shared, user = private).
#    System models are available to all users; user models are private.
umrs-client model list --scope system
umrs-client model list --scope user

# 3. Verify that an embedding model is already visible to the server
umrs-client model list --scope system
umrs-client model list --scope user

# Optional: stage a user-scoped copy when your site exposes a writable model root
umrs-client model download intfloat/e5-base-v2 --scope user --server-side \
   --staging-dir /path/to/site-user-model-root

# 4. Register a local model directory as a user-scoped model (optional)
#    Use this when you have trained or fine-tuned a model locally and
#    want to make it available to UMRS for inference.
#    The model directory should contain config.json and model weights.
umrs-client model upload ./my_custom_model \
    --name "demo-custom-model" \
    --type embedding \
    --scope user \
    --description "Custom model for lifecycle demo" \
    --force

# 5. Verify the model appears in the registry
umrs-client model list --scope user

# 6. Use the system model for inference
#    Upload a text file — the server will automatically embed it using
#    the configured embedding model.
echo "Testing the model lifecycle with a sample sentence." > /tmp/lifecycle_test.txt
umrs-client upload_file /tmp/lifecycle_test.txt user1-textfiles

# 7. Check final model inventory
umrs-client model list --scope system
umrs-client model list --scope user
```

## Model Scopes

| Scope | Who can see | Managed by | Use case |
|-------|------------|------------|----------|
| `system` | All users | Admin | Pre-trained models (e5, SAM3, HMAE) |
| `project` | Project members | Project lead | Team-trained models |
| `user` | Owner only | User | Personal experiments, fine-tuned models |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `intfloat/e5-base-v2` | HuggingFace model to download |
| `TEXT_BUCKET` | `user1-textfiles` | Bucket alias for inference test |
| `DEMO_LOCAL_MODEL_UPLOAD` | `0` | Set to `1` to enable user model upload step |

## Expected Output

```
[model_life]  System models:
[model_life]    intfloat/e5-base-v2  (embedding, system)
[model_life]  User models:
[model_life]    (none)
[model_life]   ✓ Current model inventory retrieved
[model_life]   ✓ Verified or staged an embedding model: intfloat/e5-base-v2
[model_life]   ✓ Inference triggered (server will embed automatically)

Summary of model lifecycle:
  ✓ Listed existing system and user models
   ✓ Verified or staged an embedding model: intfloat/e5-base-v2
  ✓ Triggered inference with system model
  ✓ Verified model registry state
```

## See Also

- [Demo 07: Model Registry](../07_model_registry/README.md) — download, register, list models
- [Demo 02: Text Embedding](../02_text_embedding/README.md) — full embedding pipeline
- [Demo 08: User-Defined Model](../08_user_model/README.md) — custom Step API for your own models
