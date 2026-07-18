# HMAE Full Pipeline Meeting Demo

This private shell demo exercises all three provisioned HMAE BentoML task buckets in sequence: full reconstruction, MAE reconstruction, and feature extraction. It uses the deployed HMAE experience wiring rather than uploading the bundled workflow package.

## What Runs

- `demo-hmae`: reconstruction, producing `*_recon.png` and `*_recon_results.pkl`
- `demo-hmae-mae`: masked reconstruction, producing `*_mae.png` and `*_mae_results.pkl`
- `demo-hmae-features`: feature extraction, producing `*_features.pt` and `*_features_results.pkl`
- Input images: Spring-8 concrete slices in `umrs/tmp/hmae/spring8concrete`
- Runtime service: `bentoml-vision` through the `vision-bentoml` profile
- Adapter path: `umrs/src/umrs/vision_service/default_adapters/hmae_adapter.py`

## Meeting Steps

Run from the repository root unless noted.

1. Confirm the direct UMRS profile is reachable:

   ```bash
   cd umrs
   PYTHONPATH=src:$PYTHONPATH umrs-client --profile umrs-front status
   ```

2. Run the strict one-slice three-task HMAE flow:

   ```bash
   cd demos_private/10_hmae_full_pipeline
   ```

   Then run the following commands in sequence (or copy-paste all at once):

   ```bash
   # Upload Spring-8 concrete inputs to the three experience buckets
   umrs-client data dir-upload ../../tmp/hmae/spring8concrete demo-hmae
   umrs-client data dir-upload ../../tmp/hmae/spring8concrete demo-hmae-mae
   umrs-client data dir-upload ../../tmp/hmae/spring8concrete demo-hmae-features

   # Wait for processing to complete, then download all results
   umrs-client data download demo-hmae ./results/reconstruction --force-overwrite
   umrs-client data download demo-hmae-mae ./results/mae_reconstruction --force-overwrite
   umrs-client data download demo-hmae-features ./results/feature_extraction --force-overwrite
   ```

3. Point out these validated output lines:

   ```text
   reconstruction: 2 file(s)
   mae_reconstruction: 2 file(s)
   feature_extraction: 2 file(s)
   6 total file(s) across all tasks
   ```

4. Show artifacts by task:

   ```bash
   find results -maxdepth 2 -type f -print
   ```

## Useful Overrides

- Use more slices: `MAX_IMAGES=3`
- Use another Spring-8 slice directory: `IMAGE_DIR=/path/to/slices`
- Require completed outputs: `HMAE_REQUIRE_RESULTS=1 RESULT_DOWNLOAD_TIMEOUT=2400`