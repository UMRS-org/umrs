#!/usr/bin/env python3
"""Compact chart data for the report from the result files in argv[1]."""

import json
import statistics as st
import sys
from pathlib import Path

R = Path(sys.argv[1])


def med(v):
    v = [x for x in v if x is not None]
    return round(st.median(v), 2) if v else None


def pick(rows, phase, key, field, value):
    return [r[key] for r in rows if r["phase"] == phase and r.get(field) == value]


def decode(rows, task):
    out = []
    for c in (1, 2, 4):
        sel = [r for r in rows if r["phase"] == "decode" and r["task"] == task and r["conc"] == c]
        out.append(
            [c]
            + [
                med([r[k] for r in sel])
                for k in ("decode_tok_s_per_stream_p50", "aggregate_tok_s", "itl_p50_ms", "ttft_p50_s")
            ]
        )
    return out


out = {}
for lab, f in [
    ("roce_d1", "gb10-tf-tp2-rdma-drafts1.468257.json"),
    ("roce_d0", "gb10-tf-tp2-rdma-drafts0.468257.json"),
    ("tcp_d0", "gb10-tf-tp2-tcp10g-drafts0.468257.json"),
]:
    rows = json.load(open(R / f))["rows"]
    o = {"dec_prose": decode(rows, "prose"), "dec_code": decode(rows, "code")}
    o["prefill"] = [
        [
            med(pick(rows, "prefill", k, "target_tokens", t))
            for k in ("prompt_tokens_p50", "prefill_tok_s_p50", "ttft_p50_s")
        ]
        for t in (1024, 4096, 16384, 32768, 65536, 114688)
    ]
    o["mixed"] = [
        [
            c,
            med(pick(rows, "mixed", "ttft_p50_s", "conc", c)),
            med(pick(rows, "mixed", "decode_tok_s_per_stream_p50", "conc", c)),
        ]
        for c in (1, 2, 4)
    ]
    out[lab] = o
for lab, f in [("roce", "gb10-pair-roce.468148.json"), ("tcp", "gb10-pair-tcp10g.468148.json")]:
    d = json.load(open(R / f))
    out["ar_" + lab] = [[a["bytes"], round(a["lat_us"], 1), round(a["busbw_GBps"], 2)] for a in d["allreduce"]]
    out["gpu"] = {k: round(v, 1) for k, v in d["ranks"][0].items() if isinstance(v, float)}
print(json.dumps(out, separators=(",", ":")))
