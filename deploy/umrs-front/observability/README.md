# umrs-front Observability stack (WP-12 round 2 + round 3)

The data0 (umrs-front) observability stack as deployed live:

- `compose.observability.yaml` — prom + promtail + loki + jaeger + grafana + otel-collector + node-exporter.
- `otel-collector/otel-collector-config.yaml` — OTLP receivers (gRPC :4317 / HTTP :4318), traces/logs/metrics pipelines.
- `prometheus/prometheus.yaml` — scrape config (umrs-front host-networked services via `host.containers.internal`).
- `prometheus/alerts.yaml` — rule groups: gateway / umrs / llm / infrastructure / observability / umrs-slo / **llm-slo** (second `llm` group renamed 2026-09-11: name collision caused promoter crash before this).
- `loki/promtail-config.yaml` — Loki push config.

## Deploy (umrs-front)

1. Configure OTLP loopback-only publishes (the compose file carries
   `127.0.0.1:4317/4318 + 55679 (zpages debug)` already: only loopback).
2. Bring the stack up:
   ```
   cd umrs/deploy/umrs-front/observability
   podman-compose -f compose.observability.yaml up -d
   ```
3. Wire gateway sidecars + umrs servers:
   - Each lane's `gateway-sidecar.env` gets:
     `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318`,
     `OTEL_SERVICE_NAME=umrs-front-<cluster>-gateway-sidecar`
     (sidecars run host-net: loopback OTLP http is the reachable lane;
     service deactivation needs ONLY the endpoint unset — boot is no-op).
   - umrs-server-* containers on the umrs stack's bridge network use
     `http://host.containers.internal:4318` (include them in a later
     rollout once uma [otel] extra lands in the server image).

## Learned operational classes (all pinned live 2026-09-11)

- **Jaeger badger volume**: image user is uid 10001; rootless bind-mounted
  dir must be chowned accordingly before start.
- **Prometheus volume + flags**: data dir owned uid 65534; remote-write
  receiver needs `--web.enable-remote-write-receiver` explicitly (alerts
  targets /api/v1/write gate).
- **Prom config rules**: one duplicated rule-group name makes promot fail
  with `one or more errors occurred while applying the new configuration`
  (load-error, not arg-error); no spec fix possible — rename the group,
  not the file (done locally: second `llm` -> `llm-slo`).
- **Force-recreate broken**: containers with dependents (grafana,
  otel-collector) block `podman rm` of prometheus silently; take them down
  in one wave instead.
- **Bind-mount staleness**: git pull's rename swap leaves podman binds
  pointing at the old inode — mounted configs stay byte-equivalent to old
  checkouts; `podman restart` re-resolves (or `podman cp`).
- **OTel app boot is no-op by design**: `OTEL_EXPORTER_OTLP_ENDPOINT` is
  the only SLA — with it unset the platform never starts tracing.
- **Sampling tail policy on the collector**: at probe volume its tail
  policy chain eats every sample; jaeger storage flush lag masks the chain.

## Collector configuration in hand (current)

- receive: OTLP gRPC + HTTP on 0.0.0.0:4317/4318.
- processors: memory_limiter, filter/traces (drop /health + /metrics targets), batch.
- exporters: `otlp/jaeger` (insecure, tail traces), `loki`,
  `prometheus` local exporter on :8889, `prometheusremotewrite`
  (must use the FULL container name `umrs-front-prometheus`, not
  the compose short `prometheus`).
- traces: NO tail_sampling in the round-3 patch (removed on probe, owned
  intentionally until a dedicated policy wave).

## UI endpoints

- jaeger UI: `http://<host>:50686`
- prometheus: `http://<host>:50190`
- grafana: `http://<host>:50300`
