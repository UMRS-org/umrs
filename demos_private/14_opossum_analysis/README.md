# Demo 14: Opossum Motion Analysis (Private)

This private demo wraps an opossum motion-analysis workflow for video input.
The checked-in repo contains the UMRS step wrapper and workflow spec, while the
actual external opossum analysis engine is expected to be available separately
in the local environment.

## What is in this directory

- `workflow/opossum_step.py` — the Pipeline SDK step wrapper.
- `workflow/workflow.json` — single-step workflow spec.
- `generate_test_videos.py` — synthetic test-video generator.
- `run_demo.sh` — shell entrypoint for upload, workflow upload, and result collection.

## Important Current-State Note

The script defaults to a site-local proprietary corpus path and also tries a
local fallback analysis through an external opossum engine checkout if it is
present. That external tree is **not** part of the current repo.

On the current `rccs-cloud` validation setup, the usable upload bucket is the
configured movie bucket `user1-video`, not `user1-textfiles`. The private
`opossum-analysis` workflow can still be uploaded and registered, but server-side
execution requires site wiring beyond what is present in this repo. The portable
validated path is therefore:

1. upload the workflow bundle successfully
2. optionally invoke it explicitly with `umrs-client workflow run opossum-analysis --bucket user1-video`
3. use synthetic or user-provided videos
4. run the local fallback analysis if the external opossum engine checkout is present

Automatic uploads to `user1-video` still exercise the site's configured
`video-motion-analysis` experience unless the deployment is explicitly rewired to
route that bucket through `opossum-analysis`.

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
3. Uploads the workflow from `workflow/` to the server.
4. Uploads videos to the selected bucket.
5. Attempts to download server-side results.
6. If an external opossum engine checkout is available, it also runs a local
   fallback analysis and writes `results/motion_events.tsv`.

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. (Optional) Generate synthetic test videos
#    Creates short MP4 files with simulated motion for testing.
python3 generate_test_videos.py --output-dir sample_videos/ --num-videos 4

# 3. Upload the opossum workflow to the server
#    The workflow wraps the opossum motion analysis engine as a
#    Pipeline SDK step for server-side execution via Slurm.
umrs-client upload-workflow workflow/ --name opossum-analysis

# 3b. (Optional) Explicitly invoke the uploaded workflow
umrs-client workflow run opossum-analysis --bucket user1-video

# 4. Upload videos for analysis
#    Each video is analyzed for motion events by comparing
#    consecutive frames using the opossum engine.
umrs-client upload_dir ./sample_videos user1-video

# 5. Download analysis results
#    Results include motion_events.tsv with start/end timestamps
#    for each detected motion period.
umrs-client download_results user1-video ./results --force-overwrite
```

## Defaults Used By The Script

| Setting | Default |
|---|---|
| Bucket alias | `user1-video` |
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
- [workflow/workflow.json](workflow/workflow.json)
