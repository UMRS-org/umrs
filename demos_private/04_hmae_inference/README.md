# Demo 04: HMAE Inference

Run HMAE (Hierarchical Masked Auto-Encoder) reconstruction on volume images
using the UMRS server's built-in `PROCESS_OBJECTS` / `WAIT_OUTPUT` flow.

The HMAE inference engine lives at **`tmp/hmae-infer`** in the repo root.
See [`tmp/hmae-infer/README.md`](../../tmp/hmae-infer/README.md) for the
standalone inference tool (configs, model architecture, CLI arguments).

## Prerequisites

| Requirement | How to check |
|-------------|--------------|
| UMRS server running | `umrs-client -c $CONF ping` |
| SSH tunnels active | `nc -z localhost 5555 && nc -z localhost 9000` |
| BentoML vision service running on HPC | Server log shows `Slurm job … started` for HMAE |
| HMAE checkpoint registered | `umrs-client -c $CONF model-list` shows `HMAE ViT-B` |

## Experience Config (user1)

| Field | Value |
|-------|-------|
| Experience | HMAE Inference |
| Modality | `image` (task: `reconstruction`) |
| Input bucket | `user1.hmae-inference.user1-hmae` |
| Output bucket | `user1.hmae-inference.user1-hmae-output` |
| Model | `HMAE ViT-B` via BentoML adapter (`rivault://bentoml://…`) |
| SQL table | `hmae_infer` (embeddings not stored) |

## Quick Start

```bash
cd demos/04_hmae_inference
bash run_demo.sh
```

## Step-by-step

### 1. Upload volume images to S3

Place `.png` volume slices in the input bucket:

```bash
CONF=~/.config/umrs/users/user1.json

# Upload a directory of volume slices
umrs-client -c "$CONF" upload-dir ./sample_images/ \
  --experience "HMAE Inference" \
  --modality image
```

The server auto-detects the `user1-hmae` bucket alias and uploads to
`s3://user1.hmae-inference.user1-hmae/`.

### 2. Process objects (trigger HMAE inference)

```bash
umrs-client -c "$CONF" process-objects \
  --experience "HMAE Inference"
```

This sends `PROCESS_OBJECTS` to the server, which:
1. Launches a Slurm job on a GPU node (if not already running)
2. Submits each image to the BentoML HMAE service
3. Runs quadtree decomposition → encoder → decoder (see `tmp/hmae-infer/engine/hmae.py`)
4. Uploads reconstruction PNGs to the output bucket

### 3. Wait for results

```bash
umrs-client -c "$CONF" wait-output \
  --experience "HMAE Inference"
```

Polls with exponential back-off until all images are processed.

### 4. Download results

```bash
umrs-client -c "$CONF" download-results \
  --experience "HMAE Inference" \
  -o ./results/
```

This downloads the `*_recon.png` files from the output bucket.

### 5. Check status

```bash
umrs-client -c "$CONF" status
```

## Optional: Upload HMAE Model

If the HMAE model checkpoint is not yet registered on the server:

```bash
# Register a model from the HPC filesystem
umrs-client -c "$CONF" model-register \
  --name "hmae-pretrained" \
  --path "/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/hmae" \
  --model-type embedding \
  --scope system \
  --description "HMAE ViT pretrained checkpoint (s8d-1k)"

# Or upload a local checkpoint directory
umrs-client -c "$CONF" model-upload \
  --name "hmae-pretrained" \
  --local-dir ~/Downloads/hmae_checkpoint/
```

## HMAE Inference Engine

The HMAE inference code is at **`tmp/hmae-infer/`**:

```
tmp/hmae-infer/
├── configs/infer_hmae_vit_b.yaml   # Model config (ViT-B, patch_size=32, img_size=1024)
├── engine/hmae.py                  # Quadtree decomposition engine
├── model/hvit.py                   # HMAE-ViT architecture (Base/Large/XLarge)
├── infer.py                        # Main inference entry point
├── scripts/run_infer.sh            # Quick-start shell script
└── README.md                       # Full documentation
```

### Standalone inference (outside UMRS)

```bash
cd tmp/hmae-infer
conda run -n gdt python infer.py \
  --config configs/infer_hmae_vit_b.yaml \
  --image_dir /path/to/volume_slices/ \
  --checkpoint /path/to/latest_checkpoint.pth \
  --mode reconstruct
```

Modes: `reconstruct` (full), `mae` (masked 50%), `features` (encoder-only → `.pt` files).

## Related Demos

- **Demo 03**: [SAM3 Segmentation](../03_sam3_segmentation/) — similar vision pipeline via BentoML
- **Demo 10**: [HMAE Full Pipeline](../10_hmae_full_pipeline/) — multi-step DAG: reconstruct + MAE + features in parallel
- **Demo 12**: [Pipeline SDK](../12_pipeline_sdk/) — custom Step pipelines
