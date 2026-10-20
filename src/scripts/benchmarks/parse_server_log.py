#!/usr/bin/env python3
"""
Parse UMRS server logs and extract timing information per task.
Usage: ./parse_umrs_logs.py <logfile> [--csv]
"""

import argparse
import re
import sys
from datetime import datetime
from dataclasses import dataclass, field

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"

P_AUTH        = re.compile(TS + r".*Auth session created: user=(\S+) role=")
P_PROCESS_REQ = re.compile(TS + r".*cmd=PROCESS_OBJECTS client_id=(\S+) fields=")
P_BATCH       = re.compile(TS + r".*Batch assembled:.*\('(\S+)',\s*(\d+)\)")
P_TASKS       = re.compile(TS + r".*Tasks created:.*\('(\S+)',\s*(\d+)\)")
P_SKIP        = re.compile(TS + r".*filter_already_indexed: skipping (\d+)")
P_S3_READY    = re.compile(TS + r".*S3 events ready.*\('(\S+)',\s*(\d+)\)")
P_RUNNING     = re.compile(TS + r".*\[TASK\] client=(\S+) task_id=(\d+) → Running")
P_COMPLETED   = re.compile(TS + r".*\[TASK\] client=(\S+) task_id=(\d+) → Completed")
P_TIMING      = re.compile(r"\[TIMING\] (.*)$")

# [TIMING] keys -> CSV columns (--timing)
TIMING_COLUMNS = [
    ("path", "path"),
    ("s3_events_s", "s3_events(s)"),
    ("index_check_sql_s", "index_check_sql(s)"),
    ("index_check_head_s", "index_check_head(s)"),
    ("dispatch_s", "dispatch(s)"),
    ("lock_wait_s", "lock_wait(s)"),
    ("fetch_preprocess_s", "fetch_preprocess(s)"),
    ("forward_s", "forward(s)"),
    ("embed_s", "embed(s)"),
    ("store_s", "store(s)"),
    ("count_update_s", "count_update(s)"),
    ("total_s", "server_total(s)"),
]


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")

def diff_s(t1, t2):
    if t1 and t2:
        return round((t2 - t1).total_seconds(), 2)
    return None


@dataclass
class Task:
    task_id: str
    user: str
    n_objects_batch: int = None   # from Batch assembled (total sent by client)
    n_objects_s3: int = None      # from S3 events ready (after dedup)
    n_skipped: int = 0
    t_auth: datetime = None
    t_process_req: datetime = None
    t_tasks_created: datetime = None
    t_s3_ready: datetime = None
    t_running: datetime = None
    t_completed: datetime = None
    timing: dict = field(default_factory=dict)   # key=value fields of the [TIMING] line

    @property
    def upload_s3(self):
        return diff_s(self.t_auth, self.t_process_req)

    @property
    def s3_check(self):
        return diff_s(self.t_tasks_created, self.t_s3_ready)

    @property
    def server_exec(self):
        return diff_s(self.t_running, self.t_completed)

    @property
    def end_to_end(self):
        return diff_s(self.t_process_req, self.t_completed)


def main():
    parser = argparse.ArgumentParser(description="Parse UMRS server logs for timing info.")
    parser.add_argument("logfile")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--timing", action="store_true",
                        help="add the per-step columns of the [TIMING] lines (CSV only)")
    args = parser.parse_args()

    tasks = {}            # task_id -> Task
    last_auth = {}        # user -> timestamp
    last_req = {}         # user -> timestamp
    last_batch = {}       # user -> (timestamp, n_objects)
    last_task_id = {}     # user -> task_id (most recent Tasks created)
    pending_skip = []     # skip lines before S3 ready

    with open(args.logfile) as f:
        lines = f.readlines()

    for line in lines:
        m = P_AUTH.search(line)
        if m:
            last_auth[m.group(2)] = parse_ts(m.group(1))
            continue

        m = P_PROCESS_REQ.search(line)
        if m:
            last_req[m.group(2)] = parse_ts(m.group(1))
            continue

        m = P_BATCH.search(line)
        if m:
            ts, user, n = parse_ts(m.group(1)), m.group(2), int(m.group(3))
            last_batch[user] = (ts, n)
            continue

        m = P_TASKS.search(line)
        if m:
            ts, user, task_id = parse_ts(m.group(1)), m.group(2), m.group(3)
            t = Task(task_id=task_id, user=user)
            t.t_tasks_created = ts
            t.t_auth = last_auth.get(user)
            t.t_process_req = last_req.get(user)
            if user in last_batch:
                t.n_objects_batch = last_batch[user][1]
            tasks[task_id] = t
            last_task_id[user] = task_id
            continue

        m = P_SKIP.search(line)
        if m:
            pending_skip.append(int(m.group(2)))
            continue

        m = P_S3_READY.search(line)
        if m:
            ts, user, n = parse_ts(m.group(1)), m.group(2), int(m.group(3))
            tid = last_task_id.get(user)
            if tid and tid in tasks:
                tasks[tid].t_s3_ready = ts
                tasks[tid].n_objects_s3 = n
                if pending_skip:
                    tasks[tid].n_skipped = pending_skip[-1]
                    pending_skip.clear()
            else:
                # task_id not yet seen (missing Tasks created line) — create entry
                t = Task(task_id="?", user=user)
                t.t_s3_ready = ts
                t.n_objects_s3 = n
                t.t_auth = last_auth.get(user)
                t.t_process_req = last_req.get(user)
                if pending_skip:
                    t.n_skipped = pending_skip[-1]
                    pending_skip.clear()
                # store with synthetic key
                key = f"s3_{ts.strftime('%H%M%S')}"
                tasks[key] = t
                last_task_id[user] = key
            continue

        m = P_RUNNING.search(line)
        if m:
            ts, user, task_id = parse_ts(m.group(1)), m.group(2), m.group(3)
            if task_id in tasks:
                tasks[task_id].t_running = ts
            continue

        m = P_COMPLETED.search(line)
        if m:
            ts, user, task_id = parse_ts(m.group(1)), m.group(2), m.group(3)
            if task_id in tasks:
                tasks[task_id].t_completed = ts
            continue

        m = P_TIMING.search(line)
        if m:
            fields = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
            for task_id in fields.get("task_id", "").split(","):
                if task_id in tasks:
                    tasks[task_id].timing = fields
            continue

    # Print
    header = ["task_id", "user", "n_batch", "n_skipped", "n_processed",
              "upload_s3(s)", "s3_check(s)", "server_exec(s)", "end_to_end(s)"]

    if args.csv:
        if args.timing:
            header = header + [col for _, col in TIMING_COLUMNS]
        print(",".join(header))
        for t in tasks.values():
            n_proc = (t.n_objects_s3 or 0)
            row = [
                t.task_id, t.user, t.n_objects_batch, t.n_skipped, n_proc,
                t.upload_s3, t.s3_check, t.server_exec, t.end_to_end
            ]
            if args.timing:
                row += [t.timing.get(key, "NA") for key, _ in TIMING_COLUMNS]
            print(",".join(str(x) for x in row))
    else:
        col_w = [10, 12, 9, 10, 12, 14, 12, 16, 14]
        fmt = "".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*header))
        print("-" * sum(col_w))
        for t in tasks.values():
            n_proc = (t.n_objects_s3 or 0)
            print(fmt.format(
                t.task_id, t.user,
                str(t.n_objects_batch), str(t.n_skipped), str(n_proc),
                str(t.upload_s3), str(t.s3_check),
                str(t.server_exec), str(t.end_to_end)
            ))


if __name__ == "__main__":
    main()
