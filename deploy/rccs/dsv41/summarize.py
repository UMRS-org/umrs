#!/usr/bin/env python3
"""Compact tables from serve_bench.py JSON files (median over reps)."""

import json
import statistics
import sys
from collections import defaultdict


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"== {d['meta']['label']}  ({path.split('/')[-1]})")
    groups = defaultdict(list)
    for r in d["rows"]:
        key = (r["phase"], r.get("task", ""), r.get("conc", r.get("target_tokens")))
        groups[key].append(r)
    for (ph, task, k), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        print(
            f"  {ph:7s} {task:5s} {k:>7}: ttft {fmt(med([r['ttft_p50_s'] for r in rows]), 2)} s"
            f"  dec/stream {fmt(med([r['decode_tok_s_per_stream_p50'] for r in rows]))}"
            f"  agg {fmt(med([r.get('aggregate_tok_s') for r in rows]))}"
            f"  itl {fmt(med([r['itl_p50_ms'] for r in rows]))} ms"
            f"  prefill {fmt(med([r['prefill_tok_s_p50'] for r in rows]), 0)} tok/s"
            f"  prompt {fmt(med([r['prompt_tokens_p50'] for r in rows]), 0)}"
            f"  err {sum(r['errors'] for r in rows)}"
        )
