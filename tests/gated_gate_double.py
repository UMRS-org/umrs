"""A scripted stand-in for the person's gate (ADR-112 B2 server tests).

Run as ``python gated_gate_double.py <state dir> <verb> [args...]``: the
argv the server's ``GateTransport`` builds in tests (the real transport is
``ssh ... <login>@<target> '<verb line>'``). It speaks the B2.0 §5 verb
answers and exit codes, records every verb line and every stdin it got in
``<state>/calls.jsonl``, and reads its behaviour from ``<state>/plan.json``:

- ``refuse``: ``{verb: [name, detail]}`` refuses that verb (exit 4);
- ``fail``: ``{verb: code}`` exits with that code (5 = Slurm, 255 = ssh);
- ``states``: the job states ``status`` answers in turn (last repeats);
- ``reply``: the reply object ``read-reply`` returns;
- ``out``: ``{member: text}`` that ``collect-out`` streams;
- ``nonce``: ``{"job_id", "state"}`` a ``lookup`` finds before any submit;
- ``refuse_once``: ``{verb: [name, detail]}`` refuses that verb once (wave C).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile


def _load(path: str, default):
    try:
        with open(path) as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def _save(path: str, value) -> None:
    with open(path, "w") as handle:
        json.dump(value, handle)


def _answer(obj) -> int:
    sys.stdout.write(json.dumps(obj))
    return 0


def _refuse(name: str, detail: str = "") -> int:
    sys.stderr.write(json.dumps({"error": name, "detail": detail}) + "\n")
    return 4


def _lookup(state: dict, plan: dict, intent: str, step: str) -> int:
    nonce = state["nonces"].get(f"{intent}/{step}") or plan.get("nonce")
    if not nonce:
        return 3
    return _answer({"wd": nonce.get("wd", ""), "job_id": nonce["job_id"], "submitted": 1, "state": nonce.get("state", "RUNNING")})


def _mkwd(state: dict, intent: str, step: str) -> int:
    state["seq"] += 1
    wd = f"{intent[:16]}-{step}-{state['seq']:08x}"
    state["wds"][wd] = {"intent": intent, "step": step, "members": []}
    return _answer({"wd": wd})


def _stage_in(state: dict, wd: str, data: bytes) -> int:
    members = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as tar:
        for member in tar:
            content = tar.extractfile(member).read()
            members.append([member.name, len(content)])
    state["wds"][wd]["members"] = members
    return _answer({"files": len(members), "bytes": sum(m[1] for m in members), "free_bytes": 1 << 40})


def _submit(state: dict, wd: str, step: str, data: bytes) -> int:
    envelope = json.loads(data)
    record = state["wds"][wd]
    job_id = str(4000 + state["seq"])
    state["nonces"][f"{record['intent']}/{step}"] = {"wd": wd, "job_id": job_id, "state": "PENDING"}
    state["envelopes"].append(envelope)
    return _answer({"job_id": job_id, "wd": wd, "intent_id": record["intent"], "step": step, "log_head": {"n": 7, "h": "ab" * 32}})


def _status(state: dict, plan: dict, job_id: str) -> int:
    states = plan.get("states") or ["COMPLETED"]
    index = state["polls"].get(job_id, 0)
    state["polls"][job_id] = index + 1
    return _answer({"job_id": job_id, "state": states[min(index, len(states) - 1)]})


def _collect_out(plan: dict) -> int:
    buffer = io.BytesIO()
    out = plan.get("out") or {}
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, text in out.items():
            raw = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    sys.stdout.buffer.write(buffer.getvalue())
    sys.stderr.write(json.dumps({"files": len(out), "bytes": sum(len(t) for t in out.values())}) + "\n")
    return 0


def _usage(job_id: str) -> int:
    return _answer({"jobs": [{"job_id": job_id, "state": "COMPLETED", "elapsed": "00:02:00", "alloc_tres": "cpu=1,gres/gpu=2"}]})


def _dispatch(state: dict, plan: dict, verb: str, args: list[str], data: bytes) -> int:
    simple = {
        "rmwd": lambda: _answer({"removed": True}),
        "cancel": lambda: _answer({"job_id": args[0], "cancelled": True}),
        "tail-log": lambda: _answer({"tail": "the step's last words"}),
        "read-reply": lambda: _answer(plan.get("reply") or {"status": "ok", "result": {"output_keys": []}}),
        "collect-out": lambda: _collect_out(plan),
        "usage": lambda: _usage(args[0]),
        "sweep": lambda: _answer({"removed": [], "kept": 0, "pruned": 0}),
        "hello": lambda: _answer(plan.get("hello") or {}),
    }
    if verb in simple:
        return simple[verb]()
    if verb == "lookup":
        return _lookup(state, plan, args[0], args[1])
    if verb == "mkwd":
        return _mkwd(state, args[0], args[1])
    if verb == "stage-in":
        return _stage_in(state, args[0], data)
    if verb == "submit":
        return _submit(state, args[0], args[1], data)
    if verb == "status":
        return _status(state, plan, args[0])
    return _refuse("unknown_verb", verb)


def main(argv: list[str]) -> int:
    root, verb, args = argv[1], argv[2], argv[3:]
    data = sys.stdin.buffer.read()
    plan = _load(os.path.join(root, "plan.json"), {})
    state = _load(os.path.join(root, "state.json"), {"seq": 0, "wds": {}, "nonces": {}, "envelopes": [], "polls": {}})
    with open(os.path.join(root, "calls.jsonl"), "a") as log:
        log.write(json.dumps({"line": [verb, *args], "stdin": len(data)}) + "\n")
    if verb in (plan.get("refuse") or {}):
        name, detail = plan["refuse"][verb]
        return _refuse(name, detail)
    once = plan.get("refuse_once") or {}
    if verb in once:  # wave C (K6): a refusal the next call no longer gets
        name, detail = once.pop(verb)
        _save(os.path.join(root, "plan.json"), plan)
        return _refuse(name, detail)
    fails = plan.get("fail") or {}
    if verb in fails:
        code = fails.pop(verb)
        _save(os.path.join(root, "plan.json"), plan)  # fail once
        return int(code)
    try:
        return _dispatch(state, plan, verb, args, data)
    finally:
        _save(os.path.join(root, "state.json"), state)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
