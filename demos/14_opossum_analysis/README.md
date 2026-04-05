# Demo 14: Opossum Motion Analysis

Detect opossum mating behavior from security camera videos using frame-differencing
motion analysis, running on UMRS via the Pipeline SDK.

The opossum analysis code lives at **`tmp/opossum`** in the repo root.
See [`tmp/opossum/src/opossum/analyze.py`](../../tmp/opossum/src/opossum/analyze.py) for
the core analysis engine and [`tmp/opossum/Dockerfile`](../../tmp/opossum/Dockerfile) for
the containerized runtime.

## Data

By default the demo uses **real observation footage** from `tmp/opossum/data/20260324/`:

| Subdir | Hour | Videos | Size each |
|--------|------|--------|-----------|
| `20/`  | 20h  | 5      | ~99 MB    |
| `21/`  | 21h  | 5      | ~99 MB    |
| `22/`  | 22h  | 4      | ~99 MB    |
| `23/`  | 23h  | 5      | ~99 MB    |

**Total**: 19 MP4 files, ~1.8 GB, recorded 2026-03-24 20:00–23:57.

To limit processing time, set `MAX_VIDEOS=N` (e.g. `MAX_VIDEOS=4 bash run_demo.sh`).
To use synthetic test data instead: `GEN_TEST_VIDEOS=1 bash run_demo.sh`.

## Prerequisites

| Requirement | How to check |
|-------------|--------------|
| UMRS server running | `umrs-client -c $CONF ping` |
| SSH tunnels active | `nc -z localhost 5555 && nc -z localhost 9000` |
| Opossum experience configured | user1.json has `"Opossum Analysis"` experience |
| ffmpeg available (for local testing) | `ffmpeg -version` |

## Experience Config (user1)

| Field | Value |
|-------|-------|
| Experience | Opossum Analysis |
| Modality | `video` (mp4) |
| Input bucket | `user1.opossum-analysis.user1-opossum` |
| Output bucket | `user1.opossum-analysis.user1-opossum-output` |
| Model | `pipeline://opossum_analyze:OpossumAnalyzeStep` |
| SQL table | `opossum_events` (embeddings not stored) |

## Quick Start

```bash
cd demos/14_opossum_analysis

# Use real data (default — processes all 19 videos, ~18 pairs)
bash run_demo.sh

# Limit to first N videos for a quick test
MAX_VIDEOS=4 bash run_demo.sh

# Use a custom video directory
bash run_demo.sh /path/to/videos/

# Generate and use synthetic test videos instead
GEN_TEST_VIDEOS=1 bash run_demo.sh
```

## Step-by-step

### 1. Upload videos to S3

Place `.mp4` video files in the input bucket:

```bash
CONF=~/.config/umrs/users/user1.json

# Upload a directory of opossum camera videos
umrs-client -c "$CONF" upload-dir ./sample_videos/ \
  --experience "Opossum Analysis" \
  --modality video
```

### 2. Process objects (trigger motion analysis)

```bash
umrs-client -c "$CONF" process-objects \
  --experience "Opossum Analysis"
```

This sends `PROCESS_OBJECTS` to the server, which invokes the
`OpossumAnalyzeStep` pipeline step. The step:
1. Downloads video pairs from S3
2. Runs frame-differencing motion analysis (see `tmp/opossum/src/opossum/analyze.py`)
3. Merges adjacent motion bounding boxes
4. Detects continuous motion events exceeding the time threshold
5. Uploads TSV result files to the output bucket

### 3. Wait for results

```bash
umrs-client -c "$CONF" wait-output \
  --experience "Opossum Analysis"
```

### 4. Download results

```bash
umrs-client -c "$CONF" download-results \
  --experience "Opossum Analysis" \
  -o ./results/
```

Output TSV files contain columns: `start_time, end_time, duration, bbox_x, bbox_y, bbox_w, bbox_h`.

## Opossum Analysis Engine

The opossum analysis code is at **`tmp/opossum/`**:

```
tmp/opossum/
├── Dockerfile                   # Container image (Python 3.11 + ffmpeg + OpenCV)
├── src/
│   ├── main.py                  # Batch entry point (camera-dir iterator)
│   ├── main.sh                  # Docker-wrapped batch runner
│   ├── test.sh                  # Smoke test
│   ├── opossum/
│   │   ├── analyze.py           # Core: get_duration() — frame-diff motion detection
│   │   ├── utils.py             # Image processing: binarize, noise removal, bbox merge
│   │   ├── count_non_zero_region.py
│   │   ├── moving_average.py
│   │   └── video_concatenate.py
│   ├── concat.py / concat.sh    # Video concatenation tools
│   ├── copy_paste.py / copy_paste.sh
│   └── demo_gen.py              # Generate test videos
└── data/                        # Sample data directory
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_threshold` | `0.05` | Bounding-box merge distance (fraction of frame width) |
| `time_threshold` | `15` | Minimum continuous motion duration (seconds) |
| `video_extension` | `mp4` | Also supports `avi` |

### Standalone usage (outside UMRS)

```bash
cd tmp/opossum/src

# Single pair analysis
python -m opossum.analyze video1.mp4 video2.mp4 -o output_prefix

# Batch analysis via Docker
bash main.sh
```

### Generate test videos

```bash
cd tmp/opossum/src
python demo_gen.py --output-dir ../data/ --num-videos 4
```

## Pipeline SDK Integration

The opossum analysis runs as a Pipeline SDK step:

```json
{
  "model_type": "pipeline://opossum_analyze:OpossumAnalyzeStep",
  "return_type": "file"
}
```

The `OpossumAnalyzeStep` class inherits from `Step` and implements:
- `setup(config)` — loads distance/time thresholds
- `run(ctx: StepContext)` — downloads video pairs, calls `get_duration()`, uploads TSV results
- `teardown()` — cleanup

## Related Demos

- **Demo 04**: [HMAE Inference](../04_hmae_inference/) — similar vision output pipeline
- **Demo 12**: [Pipeline SDK](../12_pipeline_sdk/) — general pipeline step documentation
- **Demo 03**: [SAM3 Segmentation](../03_sam3_segmentation/) — image processing pipeline
