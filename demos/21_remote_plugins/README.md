# Demo 21: Remote Plugin Execution (Local Simulation)

Demonstrates the **plugin lifecycle**: define plugin specs, validate them,
simulate execution, and review the audit trail. No server required — runs
entirely on your local machine.

## Plugin Architecture

```
PluginSpec → Validate → Fetch → Install → Execute → Complete
                                                       │
                                                 Audit Log
```

## Prerequisites

- Python 3.10+ with UMRS package installed (`uv pip install -e src/`)
- **No server required** — this demo runs locally

## Quick Start

```bash
cd demos/21_remote_plugins
python3 simulate_plugins.py results
```

## What This Demo Does

1. **Define plugin specs** — creates 3 plugin definitions with different
   source types (`git_repo`, `s3_wheel`, `local_path`)
2. **Validate specs** — checks required fields, source types, and entrypoint
   format for each plugin
3. **Simulate execution** — runs through the full plugin lifecycle (fetch →
   install → execute → complete) for each plugin
4. **Generate audit log** — writes a timestamped JSON audit trail of all
   operations

## Running Manually

This demo does not use `umrs-client` because it runs entirely locally
as a Python simulation. You can run it step by step:

```bash
# 1. Run the plugin simulation.
#    The script defines 3 plugins, validates them,
#    simulates execution, and writes an audit log.
python3 simulate_plugins.py results

# 2. View the generated audit log
cat results/audit_log.json

# 3. View the plugin specs
cat results/plugin_specs.json
```

## Plugin Spec Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique plugin identifier |
| `version` | Yes | Semantic version string |
| `source_type` | Yes | `git_repo`, `s3_wheel`, or `local_path` |
| `source_uri` | Yes | Path or URL to the plugin code |
| `entrypoint` | Yes | `module:ClassName` format |
| `config` | No | Key-value configuration passed to the plugin |
| `description` | No | Human-readable description |

## Plugin Source Types

| Type | Fetch Method | Example URI |
|------|-------------|-------------|
| `git_repo` | `git clone` + `pip install` | `https://github.com/org/repo.git` |
| `s3_wheel` | Download wheel from S3 | `s3://plugins/pkg-1.0-py3-none-any.whl` |
| `local_path` | Load from local filesystem | `/opt/plugins/my_plugin` |

## Plugin Lifecycle

```
validate → fetch → install → execute → complete
```

Each stage is logged with a timestamp and status. Failed plugins are
quarantined and reported in the audit log.

## Expected Output

```
=== Plugin Spec Definitions ===
  Plugin: image-embedder v1.2.0
    Source: git_repo → https://github.com/example/image-embedder.git
  Plugin: pdf-parser v0.9.0
    Source: s3_wheel → s3://plugins/pdf_parser-0.9.0-py3-none-any.whl
  Plugin: local-analyzer v0.1.0
    Source: local_path → /opt/plugins/local_analyzer

=== Plugin Validation ===
  ✓ [validate] image-embedder: ok
  ✓ [validate] pdf-parser: ok
  ✓ [validate] local-analyzer: ok

=== Audit Log Summary ===
  Total entries : 15
  Successful    : 15
  Errors        : 0
```

## See Also

- [Demo 13: Plugins](../13_plugins/README.md) — plugin execution system (with Slurm/K8s backends)
- [Demo 08: User-Defined Model](../08_user_model/README.md) — Step API for in-process pipelines
- [Demo 16: Custom Workflow](../16_custom_workflow/README.md) — server-side workflow execution
