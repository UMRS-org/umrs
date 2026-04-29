# scientific-figure-understanding

A reference workflow that demonstrates the **multimodal perception →
LLM reasoning** pattern using the built-in step templates shipped with
`umrs.pipeline_sdk.templates`.

## Steps

1. `vision_dense` — `VisionStep(mode=dense)` produces a long
   description via the platform VLM endpoint.
2. `vision_ocr` — `OCRStep` extracts every visible text snippet.
3. `reason` — `LLMStep` renders a Jinja-style prompt against the
   upstream perception and asks the text LLM to return structured JSON
   with `{title, axes, finding, caveats}`.

## Trigger

```bash
r4 workflow upload demos/workflows/scientific-figure-understanding
r4 workflow trigger scientific-figure-understanding \
    --param image=s3://bucket/figure.png \
    --wait
```

When `UMRS_VISION_MOCK=1` and `UMRS_LLM_MOCK=1` are set the workflow
runs end-to-end without any external service — used by E2E tests.

## Routing hints

Each step declares `requires` and `model_type` so the in-process
**StepRouter** can resolve the right endpoint per step instead of
sending every request to a single shared model:

* `vision_dense` / `vision_ocr` — `requires: ["vision"]`,
  `model_type: "vision"` → routed to a vision-capable endpoint
  (Qwen2.5-VL or similar).
* `reason` — `requires: ["tool_calling", "structured_output"]`,
  `model_type: "llm"` → routed to a text LLM endpoint.

The router records its decision in `record.results['_routing'][<step>]`
for post-mortem visibility (`{api_base, model, endpoint_id,
resolved_via}`). When the model registry has no matching model the
runner fails the step early with a `StepRoutingError` rather than
silently sending the call to the wrong endpoint.

To bypass routing in tests / mock runs, set `UMRS_STEP_ROUTER_MOCK=1`;
every step then receives a static mock decision.

## Parallel execution

`vision_dense` and `vision_ocr` declare no `depends_on` between them,
so the workflow runner places them in the same wave and executes them
concurrently on a thread pool. The `reason` step depends on both, so
it waits for the wave barrier before starting. No workflow.json
change is required to opt in — the parallelism is automatic from the
DAG shape. Cap the pool with `UMRS_RUNNER_MAX_PARALLEL_STEPS` (default
8); set it to `1` to force serial execution for debugging.

