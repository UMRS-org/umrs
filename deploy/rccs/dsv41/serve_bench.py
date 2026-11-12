#!/usr/bin/env python3
"""Serving benchmark for OpenAI-compatible chat endpoints (stdlib only).

Same client, same prompts, same settings against every endpoint, so differences
come from the serving side (architecture, GPU, interconnect, engine). Phases:

  decode   concurrency sweep (1, 2, 4, ...) on short prompts: TTFT, inter-token
           latency, per-stream and aggregate decode tok/s
  prefill  one request at a time with prompts of growing length: TTFT and
           prefill tok/s (prompt tokens / TTFT), max_tokens 1
  mixed    N concurrent streams with mid-length prompts (agent-like turns)

Prompts are randomized per request (a unique nonce first) so no prefix cache
hits. Thinking is off and temperature 0 unless --thinking. Output: one JSON
file with every request's timings plus per-phase summaries.

Usage: serve_bench.py --url http://host:port --model NAME --key-file F --label L --out F.json
       [--phases decode,prefill,mixed] [--conc 1,2,4] [--lengths 1024,4096,...]
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import socket
import statistics
import threading
import time
import urllib.parse
from pathlib import Path

WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his from at which but have an they you "
    "were her she there been one all we their has would when if so no will more can out other about time up only "
    "model memory network latency bandwidth tensor parallel rank kernel cache token prefill decode stream batch "
    "science protein lattice spectrum galaxy neutron enzyme climate ocean reactor plasma quantum orbit fossil"
).split()

TASKS = {
    "prose": "Write a detailed, well-structured essay about the history of scientific computing. Keep writing.",
    "code": "Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL, "
    "with unit tests. Keep writing code.",
}


def filler(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


class Endpoint:
    def __init__(self, url: str, model: str, key: str, timeout: float) -> None:
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname or "", u.port or 80
        self.model, self.key, self.timeout = model, key, timeout

    def stream(self, messages: list[dict], max_tokens: int, thinking: bool, temperature: float) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking},
        }
        if not thinking:
            body["reasoning_effort"] = "none"
        return self._run(json.dumps(body).encode())

    def _run(self, payload: bytes) -> dict:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        t0 = time.perf_counter()
        rec: dict = {"t_send": time.time(), "chunks": 0, "token_times": [], "error": None}
        try:
            conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {resp.read(300).decode(errors='replace')}"
                return rec
            self._consume(resp, rec, t0)
        except (OSError, socket.timeout, http.client.HTTPException) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        rec["t_total"] = time.perf_counter() - t0
        return rec

    @staticmethod
    def _consume(resp: http.client.HTTPResponse, rec: dict, t0: float) -> None:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                rec["usage"] = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    rec["token_times"].append(time.perf_counter() - t0)
                    rec["chunks"] += 1


def summarize_request(rec: dict) -> dict:
    tt = rec.get("token_times") or []
    usage = rec.get("usage") or {}
    out_tokens = usage.get("completion_tokens") or rec.get("chunks", 0)
    s = {
        "error": rec.get("error"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "ttft_s": tt[0] if tt else None,
        "total_s": rec.get("t_total"),
        "chunks": rec.get("chunks", 0),
    }
    if len(tt) > 1 and out_tokens and out_tokens > 1:
        decode_time = tt[-1] - tt[0]
        s["decode_tok_s"] = (out_tokens - 1) / decode_time if decode_time > 0 else None
        gaps = [b - a for a, b in zip(tt, tt[1:], strict=False)]
        s["itl_p50_ms"] = 1000 * statistics.median(gaps)
        s["itl_p95_ms"] = 1000 * sorted(gaps)[int(0.95 * (len(gaps) - 1))]
    if s["ttft_s"] and s["prompt_tokens"]:
        s["prefill_tok_s"] = s["prompt_tokens"] / s["ttft_s"]
    return s


def run_concurrent(ep: Endpoint, jobs: list[tuple[list[dict], int]], thinking: bool, temp: float) -> list[dict]:
    results: list[dict] = [{} for _ in jobs]
    start = threading.Barrier(len(jobs))

    def worker(i: int) -> None:
        msgs, mt = jobs[i]
        start.wait()
        results[i] = ep.stream(msgs, mt, thinking, temp)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(jobs))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    for r in results:
        r["wall_s"] = wall
    return results


def agg(reqs: list[dict]) -> dict:
    ok = [r for r in reqs if not r.get("error")]

    def med(key: str):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    return {
        "n": len(reqs),
        "errors": len(reqs) - len(ok),
        "ttft_p50_s": med("ttft_s"),
        "decode_tok_s_per_stream_p50": med("decode_tok_s"),
        "itl_p50_ms": med("itl_p50_ms"),
        "itl_p95_ms": med("itl_p95_ms"),
        "prefill_tok_s_p50": med("prefill_tok_s"),
        "prompt_tokens_p50": med("prompt_tokens"),
        "completion_tokens_sum": sum(r.get("completion_tokens") or 0 for r in ok),
    }


def phase_decode(ep: Endpoint, args, rng: random.Random) -> list[dict]:
    rows = []
    for conc in args.conc:
        for task, text in TASKS.items():
            for rep in range(args.reps):
                jobs = [
                    ([{"role": "user", "content": f"[{rng.random():.12f}] {text}"}], args.decode_tokens)
                    for _ in range(conc)
                ]
                raw = run_concurrent(ep, jobs, args.thinking, args.temperature)
                reqs = [summarize_request(r) for r in raw]
                wall = raw[0]["wall_s"]
                row = {"phase": "decode", "task": task, "conc": conc, "rep": rep, **agg(reqs)}
                row["aggregate_tok_s"] = row["completion_tokens_sum"] / wall if wall else None
                row["requests"] = reqs
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
    return rows


def phase_prefill(ep: Endpoint, args, rng: random.Random) -> list[dict]:
    rows = []
    for n in args.lengths:
        for rep in range(args.reps):
            words = int(n * 0.75)  # ~1.33 tokens a word for this vocabulary; the server's count is what we report
            msg = f"[{rng.random():.12f}] Summarize in one word: " + filler(words, rng)
            raw = ep.stream([{"role": "user", "content": msg}], 1, False, 0.0)
            req = summarize_request(raw)
            row = {"phase": "prefill", "target_tokens": n, "rep": rep, **agg([req]), "requests": [req]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
    return rows


def phase_mixed(ep: Endpoint, args, rng: random.Random) -> list[dict]:
    rows = []
    for conc in args.conc:
        for rep in range(args.reps):
            jobs = [
                (
                    [
                        {
                            "role": "user",
                            "content": f"[{rng.random():.12f}] Context: {filler(args.mixed_words, rng)}\n"
                            "Explain what the context above could be about, in detail.",
                        }
                    ],
                    args.mixed_tokens,
                )
                for _ in range(conc)
            ]
            raw = run_concurrent(ep, jobs, args.thinking, args.temperature)
            reqs = [summarize_request(r) for r in raw]
            wall = raw[0]["wall_s"]
            row = {"phase": "mixed", "conc": conc, "rep": rep, **agg(reqs)}
            row["aggregate_tok_s"] = row["completion_tokens_sum"] / wall if wall else None
            row["requests"] = reqs
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="decode,prefill,mixed")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--lengths", default="1024,4096,16384,32768,65536,114688")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--mixed-words", type=int, default=6000)
    ap.add_argument("--mixed-tokens", type=int, default=256)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--meta", default="{}", help="JSON merged into the output's meta")
    args = ap.parse_args()
    args.conc = [int(x) for x in args.conc.split(",")]
    args.lengths = [int(x) for x in args.lengths.split(",")]
    ep = Endpoint(args.url, args.model, Path(args.key_file).read_text().strip(), args.timeout)
    rng = random.Random(20261112)
    warm = ep.stream([{"role": "user", "content": "Say hello."}], 16, False, 0.0)
    print("warmup", summarize_request(warm), flush=True)
    rows = []
    for ph in args.phases.split(","):
        rows += {"decode": phase_decode, "prefill": phase_prefill, "mixed": phase_mixed}[ph](ep, args, rng)
    meta = {
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "client": socket.gethostname(),
        "thinking": args.thinking,
        "temperature": args.temperature,
        "decode_tokens": args.decode_tokens,
        **json.loads(args.meta),
    }
    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
