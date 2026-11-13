#!/usr/bin/env python3
"""
Parse UMRS server logs and extract timing information per task.
Usage: ./parse_server_log.py <logfile> [--csv]

Reads the server's plain-text log (``TEXT_LOG_FORMAT`` in
umrs.server.structured_logging). Upstream's CSV also had columns for a
``[TIMING]`` line that no UMRS server writes, so they were always NA and are
gone (review U1).

One batch logs "Batch assembled" (every client request, in order), "Tasks
created" (the task ids, same order), then at most one "skipping N" line
("per client: [...]" names each client's share) and "S3 events ready" or
"All objects already indexed". Every (client, n) tuple is read; a skip
count belongs to the batch it is logged in. A skip line without the
per-client part (older servers) is attributed only when its batch has a
single client, or when no batch is open (upstream logs).
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"

P_AUTH        = re.compile(TS + r".*Auth session created: user=(\S+) role=")
P_PROCESS_REQ = re.compile(TS + r".*cmd=PROCESS_OBJECTS client_id=(\S+) fields=")
P_BATCH       = re.compile(TS + r".*Batch assembled:(.*)")
P_TASKS       = re.compile(TS + r".*Tasks created:(.*)")
# This server logs "[PROCESS] skipping N already-indexed object(s): {...}
# per client: [...]" (umrs.server.indexed_prefilter); upstream logged
# "filter_already_indexed: skipping N", still read so older logs parse.
P_SKIP        = re.compile(TS + r".*(?:\[PROCESS\]|filter_already_indexed:) skipping (\d+)(.*)")
P_ALL_INDEXED = re.compile(TS + r".*All objects already indexed")
P_S3_READY    = re.compile(TS + r".*S3 events ready(.*)")
P_RUNNING     = re.compile(TS + r".*\[TASK\] client=(\S+) task_id=(\d+) → Running")
P_COMPLETED   = re.compile(TS + r".*\[TASK\] client=(\S+) task_id=(\d+) → Completed")
P_PAIR        = re.compile(r"\('([^']+)',\s*(\d+)\)")

HEADER = ["task_id", "user", "n_batch", "n_skipped", "n_processed",
          "upload_s3(s)", "s3_check(s)", "server_exec(s)", "end_to_end(s)"]
TABLE_COL_WIDTHS = [10, 12, 9, 10, 12, 14, 12, 16, 14]


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")  # noqa: DTZ007 — server log stamps are naive local time


def pairs(text):
    """Every ``('client', n)`` tuple in *text*, in order."""
    return [(client, int(n)) for client, n in P_PAIR.findall(text)]


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
        self.last_batch = {}       # user -> n_objects (latest Batch assembled)
        self.batch_counts = []     # [(user, n)] of the latest Batch assembled, in order
        self.last_task_id = {}     # user -> task_id (most recent Tasks created)
        self.open_batch = {}       # user -> first task_id of the batch being collected
        self.pending_skip = None   # skip count logged while no batch is open
        self.handlers = [
            (P_AUTH, self._on_auth),
            (P_PROCESS_REQ, self._on_process_req),
            (P_BATCH, self._on_batch),
            (P_TASKS, self._on_tasks_created),
            (P_SKIP, self._on_skip),
            (P_ALL_INDEXED, self._close_batch),
            (P_S3_READY, self._on_s3_ready),
            (P_RUNNING, self._on_running),
            (P_COMPLETED, self._on_completed),
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
        self.batch_counts = pairs(m.group(2))
        self.last_batch.update(self.batch_counts)

    def _new_task(self, task_id, user):
        t = Task(task_id=task_id, user=user)
        t.t_auth = self.last_auth.get(user)
        t.t_process_req = self.last_req.get(user)
        return t

    def _batch_count(self, index, user):
        """The request's own count: Batch assembled and Tasks created list the
        same requests in the same order."""
        if index < len(self.batch_counts) and self.batch_counts[index][0] == user:
            return self.batch_counts[index][1]
        return self.last_batch.get(user)

    def _on_tasks_created(self, m):
        ts = parse_ts(m.group(1))
        self._close_batch(m)
        for index, (user, tid) in enumerate(pairs(m.group(2))):
            task_id = str(tid)
            t = self._new_task(task_id, user)
            t.t_tasks_created = ts
            t.n_objects_batch = self._batch_count(index, user)
            self.tasks[task_id] = t
            self.last_task_id[user] = task_id
            self.open_batch.setdefault(user, task_id)
        self.batch_counts = []

    def _close_batch(self, _m=None):
        self.open_batch = {}
        self.pending_skip = None

    def _on_skip(self, m):
        total, per_client = int(m.group(2)), pairs(m.group(3).partition("per client:")[2])
        if not self.open_batch:
            self.pending_skip = total
        elif per_client:
            self._assign_skips(per_client)
        elif len(self.open_batch) == 1:
            self._assign_skips([(next(iter(self.open_batch)), total)])

    def _assign_skips(self, per_client):
        for user, n in per_client:
            task = self.tasks.get(self.open_batch.get(user))
            if task is not None:
                task.n_skipped = n

    def _s3_task(self, user, ts):
        tid = self.open_batch.get(user) or self.last_task_id.get(user)
        if tid in self.tasks:
            return self.tasks[tid]
        # task_id not yet seen (missing Tasks created line): store with a synthetic key
        task = self._new_task("?", user)
        key = f"s3_{ts.strftime('%H%M%S')}"
        self.tasks[key] = task
        self.last_task_id[user] = key
        return task

    def _on_s3_ready(self, m):
        ts = parse_ts(m.group(1))
        for user, n in pairs(m.group(2)):
            task = self._s3_task(user, ts)
            task.t_s3_ready = ts
            task.n_objects_s3 = n
            if self.pending_skip is not None:
                task.n_skipped, self.pending_skip = self.pending_skip, None
        self._close_batch()

    def _on_running(self, m):
        task = self.tasks.get(m.group(3))
        if task is not None:
            task.t_running = parse_ts(m.group(1))

    def _on_completed(self, m):
        task = self.tasks.get(m.group(3))
        if task is not None:
            task.t_completed = parse_ts(m.group(1))


def parse_lines(lines):
    """Return the tasks (insertion-ordered dict) found in *lines*."""
    parser = LogParser()
    for line in lines:
        parser.feed(line)
    return parser.tasks


def write_csv(tasks, out):
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(HEADER)
    for t in tasks.values():
        writer.writerow(t.summary_row())


def write_table(tasks, out):
    fmt = "".join(f"{{:<{w}}}" for w in TABLE_COL_WIDTHS)
    print(fmt.format(*HEADER), file=out)
    print("-" * sum(TABLE_COL_WIDTHS), file=out)
    for t in tasks.values():
        print(fmt.format(*(str(v) for v in t.summary_row())), file=out)


def main():
    parser = argparse.ArgumentParser(description="Parse UMRS server logs for timing info.")
    parser.add_argument("logfile")
    parser.add_argument("--csv", action="store_true", help="CSV output")
    args = parser.parse_args()

    with open(args.logfile) as f:
        tasks = parse_lines(f.readlines())

    if args.csv:
        write_csv(tasks, sys.stdout)
    else:
        write_table(tasks, sys.stdout)


if __name__ == "__main__":
    main()
