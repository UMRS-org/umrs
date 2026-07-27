# UMRS Demo Day Commands

Run from `umrs/`. Set `PROFILE=rccs-cloud` once.

```bash
export PROFILE=rccs-cloud
python3 src/client_umrs.py --profile "$PROFILE" ping
python3 src/client_umrs.py --profile "$PROFILE" doctor
python3 src/client_umrs.py --profile "$PROFILE" --output json bucket list
```

## Text RAG input
```bash
python3 src/client_umrs.py --profile "$PROFILE" data dir-upload demos/sample_data/texts mnagaso3-text
python3 src/client_umrs.py --profile "$PROFILE" status all
```

## SAM3.1 segmentation input
```bash
mkdir -p /tmp/umrs-demo-image-input /tmp/umrs-defect-results
cp demos/sample_data/images/Cat_1.jpg /tmp/umrs-demo-image-input/
python3 src/client_umrs.py --profile "$PROFILE" data dir-upload /tmp/umrs-demo-image-input demo-defect
python3 src/client_umrs.py --profile "$PROFILE" status all
python3 src/client_umrs.py --profile "$PROFILE" data download demo-defect /tmp/umrs-defect-results --force-overwrite
```

## HMAE reconstruction input
```bash
mkdir -p /tmp/umrs-hmae-results
python3 src/client_umrs.py --profile "$PROFILE" data dir-upload /tmp/umrs-demo-image-input demo-hmae
python3 src/client_umrs.py --profile "$PROFILE" status all
python3 src/client_umrs.py --profile "$PROFILE" data download demo-hmae /tmp/umrs-hmae-results --force-overwrite
```

## Provision a private experience when absent
```bash
python3 src/client_umrs.py --profile "$PROFILE" experience add demos_private/04_hmae_inference/experience.json --update
python3 src/client_umrs.py --profile "$PROFILE" reload
python3 src/client_umrs.py --profile "$PROFILE" --output json bucket list
```

`data download` retrieves the experience output bucket and requires the target directory to exist. An empty output directory plus `status all` showing idle is an execution-trigger investigation, not a successful inference run.
