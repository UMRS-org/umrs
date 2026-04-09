# Demo 14: Opossum Motion Analysis (Private)

This private demo wraps an opossum motion-analysis workflow for video input.
The checked-in repo contains the UMRS step wrapper and workflow spec, while the
actual external opossum analysis engine is expected to be available separately
in the local environment.

## What is in this directory

- `opossum_workflow/opossum_step.py` — the Pipeline SDK step wrapper.
- `opossum_workflow/workflow.json` — single-step workflow spec.
- `generate_test_videos.py` — synthetic test-video generator.
- `run_demo.sh` — shell entrypoint for upload, workflow upload, and result collection.

## Important Current-State Note

The script defaults to a site-local proprietary corpus path and also tries a
local fallback analysis through an external opossum engine checkout if it is
present. That external tree is **not** part of the current repo.

For a portable smoke test, use synthetic videos instead of relying on the
missing proprietary dataset:

```bash
cd demos_private/14_opossum_analysis
GEN_TEST_VIDEOS=1 bash run_demo.sh
```

To use your own real videos:

```bash
bash run_demo.sh /path/to/videos
```

## What the script does

1. Runs the standard UMRS pre-flight checks.
2. Optionally generates synthetic `.mp4` files.
3. Uploads the workflow from `opossum_workflow/` to the server.
4. Uploads videos to the selected bucket.
5. Attempts to download server-side results.
6. If an external opossum engine checkout is available, it also runs a local
   fallback analysis and writes `results/motion_events.tsv`.

## Defaults Used By The Script

| Setting | Default |
|---|---|
| Bucket alias | `user1-textfiles` |
| Default video dir | Site-local external corpus path configured in the script |
| Synthetic mode | `GEN_TEST_VIDEOS=1` |
| Result dir | `results/` |

## External Dependency

`opossum_step.py` imports `opossum.analyze.get_duration`. That module must be
available in the external engine tree for the local fallback analysis path to
work.

## Public Analogue

The public replacement for a generic video-analysis workflow is:

- [../../demos/14_video_analysis/README.md](../../demos/14_video_analysis/README.md)

## Related Docs

- [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
- [opossum_workflow/workflow.json](opossum_workflow/workflow.json)
