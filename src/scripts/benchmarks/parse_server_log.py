#!/usr/bin/env python3
"""
Parse UMRS server logs and extract timing information per task.
Usage: ./parse_server_log.py <logfile> [--csv]
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime

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

# [TIMING] keys -> CSV columns ("NA" when the log has no [TIMING] line for the task)
TIMING_COLUMNS = [
    ("status", "status"),
    ("model", "model"),
    ("path", "path"),
    ("batch_tasks", "batch_tasks"),
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

HEADER = ["task_id", "user", "n_batch", "n_skipped", "n_processed",
          "upload_s3(s)", "s3_check(s)", "server_exec(s)", "end_to_end(s)"]
TABLE_COL_WIDTHS = [10, 12, 9, 10, 12, 14, 12, 16, 14]


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")  # noqa: DTZ007 — server log stamps are naive local time


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

    def summary_row(self):
        n_proc = (self.n_objects_s3 or 0)
        return [self.task_id, self.user, self.n_objects_batch, self.n_skipped, n_proc,
                self.upload_s3, self.s3_check, self.server_exec, self.end_to_end]


class LogParser:
    """Line-by-line state machine; the first matching pattern handles a line."""

    def __init__(self):
        self.tasks = {}            # task_id -> Task
        self.last_auth = {}        # user -> timestamp
        self.last_req = {}         # user -> timestamp
        self.last_batch = {}       # user -> (timestamp, n_objects)
        self.last_task_id = {}     # user -> task_id (most recent Tasks created)
        self.pending_skip = []     # skip lines before S3 ready
        self.handlers = [
            (P_AUTH, self._on_auth),
            (P_PROCESS_REQ, self._on_process_req),
            (P_BATCH, self._on_batch),
            (P_TASKS, self._on_tasks_created),
            (P_SKIP, self._on_skip),
            (P_S3_READY, self._on_s3_ready),
            (P_RUNNING, self._on_running),
            (P_COMPLETED, self._on_completed),
            (P_TIMING, self._on_timing),
        ]

    def feed(self, line):
        for pattern, handler in self.handlers:
            m = pattern.search(line)
            if m:
                handler(m)
                return

    def _on_auth(self, m):
        self.last_auth[m.group(2)] = parse_ts(m.group(1))

    def _on_process_req(self, m):
        self.last_req[m.group(2)] = parse_ts(m.group(1))

    def _on_batch(self, m):
        self.last_batch[m.group(2)] = (parse_ts(m.group(1)), int(m.group(3)))

    def _new_task(self, task_id, user):
        t = Task(task_id=task_id, user=user)
        t.t_auth = self.last_auth.get(user)
        t.t_process_req = self.last_req.get(user)
        return t

    def _on_tasks_created(self, m):
        ts, user, task_id = parse_ts(m.group(1)), m.group(2), m.group(3)
        t = self._new_task(task_id, user)
        t.t_tasks_created = ts
        if user in self.last_batch:
            t.n_objects_batch = self.last_batch[user][1]
        self.tasks[task_id] = t
        self.last_task_id[user] = task_id

    def _on_skip(self, m):
        self.pending_skip.append(int(m.group(2)))

    def _apply_pending_skip(self, task):
        if self.pending_skip:
            task.n_skipped = self.pending_skip[-1]
            self.pending_skip.clear()

    def _on_s3_ready(self, m):
        ts, user, n = parse_ts(m.group(1)), m.group(2), int(m.group(3))
        tid = self.last_task_id.get(user)
        if tid and tid in self.tasks:
            task = self.tasks[tid]
        else:
            # task_id not yet seen (missing Tasks created line): store with a synthetic key
            task = self._new_task("?", user)
            key = f"s3_{ts.strftime('%H%M%S')}"
            self.tasks[key] = task
            self.last_task_id[user] = key
        task.t_s3_ready = ts
        task.n_objects_s3 = n
        self._apply_pending_skip(task)

    def _on_running(self, m):
        task = self.tasks.get(m.group(3))
        if task is not None:
            task.t_running = parse_ts(m.group(1))

    def _on_completed(self, m):
        task = self.tasks.get(m.group(3))
        if task is not None:
            task.t_completed = parse_ts(m.group(1))

    def _on_timing(self, m):
        fields = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
        for task_id in fields.get("task_id", "").split(","):
            if task_id in self.tasks:
                self.tasks[task_id].timing = fields


def parse_lines(lines):
    """Return the tasks (insertion-ordered dict) found in *lines*."""
    parser = LogParser()
    for line in lines:
        parser.feed(line)
    return parser.tasks


def write_csv(tasks, out):
    writer = csv.writer(out, lineterminator="\n")  # quotes batch_tasks="1,2"
    writer.writerow(HEADER + [col for _, col in TIMING_COLUMNS])
    for t in tasks.values():
        writer.writerow(t.summary_row() + [t.timing.get(key, "NA") for key, _ in TIMING_COLUMNS])


def write_table(tasks, out):
    fmt = "".join(f"{{:<{w}}}" for w in TABLE_COL_WIDTHS)
    print(fmt.format(*HEADER), file=out)
    print("-" * sum(TABLE_COL_WIDTHS), file=out)
    for t in tasks.values():
        print(fmt.format(*(str(v) for v in t.summary_row())), file=out)


def main():
    parser = argparse.ArgumentParser(description="Parse UMRS server logs for timing info.")
    parser.add_argument("logfile")
    parser.add_argument("--csv", action="store_true",
                        help="CSV output, including every field of the [TIMING] lines")
    parser.add_argument("--timing", action="store_true",
                        help="no effect: the [TIMING] columns are always in the CSV")
    args = parser.parse_args()

    with open(args.logfile) as f:
        tasks = parse_lines(f.readlines())

    if args.csv:
        write_csv(tasks, sys.stdout)
    else:
        write_table(tasks, sys.stdout)


if __name__ == "__main__":
    main()
