#!/usr/bin/env python3
"""
Parse UMRS server logs and extract timing information per task.
Usage: ./parse_server_log.py <logfile> [--csv]

Reads the server's plain-text log (``TEXT_LOG_FORMAT`` in
umrs.server.structured_logging). Upstream's CSV also had columns for a
``[TIMING]`` line that no UMRS server writes, so they were always NA and are
gone (review U1).

One batch logs "Batch assembled" (every client request, in order) and "Tasks
created" (the task ids, same order) on the event loop. Its own collector
thread logs, up to 60 s later, at most one "skipping N" line ("per client:
[...]" names each client's share), then "S3 events ready" or "All objects
already indexed". With several clients those lines interleave with the next
batches' "Tasks created" (review U1), so the parser keeps, per client, the
tasks still waiting for their collector, oldest first: a client's skip
share goes to its oldest waiting task that has none yet, and "S3 events
ready" resolves each listed client's oldest waiting task. A task whose skip
share covers every object its client submitted is resolved there (no S3
line follows); "All objects already indexed" resolves what the latest skip
line assigned when the submitted count is unknown (no "Batch assembled").
A collector that gets no S3 event logs "Received 0/N S3 events for
client=X" or "No data queue for client=X" and no skip or S3 line: that
resolves X's oldest waiting task. Any "[TASK] client=X task_id=N → status"
line (Running, Completed, Error, ...) resolves task N, which the collector
is done with; without that a failed or fully skipped batch stayed open and
took the client's next batch's lines (review U1 r4). Every (client, n)
tuple is read. A skip line without the
per-client part (older servers) is attributed only when exactly one task
waits for a skip, or kept for the next S3 line when no task waits at all
(upstream logs). The server logs no batch id, so two batches of the same
client whose collectors finish out of order can still swap their shares.
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
P_NO_EVENTS   = re.compile(TS + r".*(?:Received 0/\d+ S3 events|No data queue) for client=(\S+)")
P_STATUS      = re.compile(TS + r".*\[TASK\] client=(\S+) task_id=(\d+) → (\S+)")
P_PAIR        = re.compile(r"\('([^']+)',\s*(\d+)\)")

HEADER = ["task_id", "user", "n_batch", "n_skipped", "n_processed",
          "upload_s3(s)", "s3_check(s)", "server_exec(s)", "end_to_end(s)"]
TABLE_COL_WIDTHS = [10, 12, 9, 10, 12, 14, 12, 16, 14]
STATUS_TIMES = {"Running": "t_running", "Completed": "t_completed"}


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f")  # noqa: DTZ007 — server log stamps are naive local time


def pairs(text):
    """Every ``('client', n)`` tuple in *text*, in order."""
    return [(client, int(n)) for client, n in P_PAIR.findall(text)]


def _plus(a, b):
    """``a + b``, or None when either is unknown."""
    if a is None or b is None:
        return None
    return a + b


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
        self.open_tasks = {}       # user -> [task_id, ...] waiting for their collector, oldest first
        self.submitted = {}        # open task_id -> objects its client sent in that batch
        self.skip_seen = set()     # open task ids that already got their skip share
        self.last_skip = []        # [(user, task_id)] the latest skip line assigned
        self.pending_skip = None   # skip count logged while no task is open
        self.handlers = [
            (P_AUTH, self._on_auth),
            (P_PROCESS_REQ, self._on_process_req),
            (P_BATCH, self._on_batch),
            (P_TASKS, self._on_tasks_created),
            (P_SKIP, self._on_skip),
            (P_ALL_INDEXED, self._on_all_indexed),
            (P_S3_READY, self._on_s3_ready),
            (P_NO_EVENTS, self._on_no_events),
            (P_STATUS, self._on_status),
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

    def _created_task(self, index, user, tid, ts):
        task = self._new_task(str(tid), user)
        task.t_tasks_created = ts
        task.n_objects_batch = self._batch_count(index, user)
        self.tasks[task.task_id] = task
        self.last_task_id[user] = task.task_id
        return task

    def _on_tasks_created(self, m):
        """Open each client's first task of the batch; earlier batches stay
        open, since their collectors may still be logging (review U1)."""
        ts = parse_ts(m.group(1))
        firsts, totals = {}, {}
        for index, (user, tid) in enumerate(pairs(m.group(2))):
            task = self._created_task(index, user, tid, ts)
            firsts.setdefault(user, task.task_id)
            totals[user] = _plus(totals.get(user, 0), task.n_objects_batch)
        for user, task_id in firsts.items():
            self.open_tasks.setdefault(user, []).append(task_id)
            self.submitted[task_id] = totals[user]
        self.batch_counts = []
        self.pending_skip = None

    def _resolve(self, user, task_id):
        """*task_id*'s collector is done: it waits for no more lines."""
        waiting = self.open_tasks.get(user, [])
        if task_id in waiting:
            waiting.remove(task_id)
        self.skip_seen.discard(task_id)
        self.submitted.pop(task_id, None)

    def _on_skip(self, m):
        total, per_client = int(m.group(2)), pairs(m.group(3).partition("per client:")[2])
        if not per_client:
            per_client = self._unnamed_skip(total)
        assigned = [(user, self._assign_skip(user, n)) for user, n in per_client]
        self.last_skip = [(user, tid) for user, tid in assigned if tid is not None]

    def _unnamed_skip(self, total):
        """A skip line without its per-client part (older servers)."""
        waiting = [(user, tid) for user, tids in self.open_tasks.items()
                   for tid in tids if tid not in self.skip_seen]
        if not any(self.open_tasks.values()):
            self.pending_skip = total
        return [(waiting[0][0], total)] if len(waiting) == 1 else []

    def _assign_skip(self, user, n):
        """Give *n* to *user*'s oldest waiting task without a skip share yet."""
        tid = next((t for t in self.open_tasks.get(user, ()) if t not in self.skip_seen), None)
        if tid is None:
            return None
        self.tasks[tid].n_skipped = n
        self.skip_seen.add(tid)
        submitted = self.submitted.get(tid)
        if submitted is not None and n >= submitted:
            self._resolve(user, tid)  # fully skipped: no S3 line will name it
        return tid

    def _on_all_indexed(self, _m):
        """Resolve only skips whose share could not be checked against the
        submitted count: a known partial share is another collector's task
        still waiting for its S3 line (review U1 r4)."""
        for user, tid in self.last_skip:
            if self.submitted.get(tid) is None:
                self._resolve(user, tid)
        self.last_skip = []
        self.pending_skip = None

    def _s3_task(self, user, ts):
        waiting = self.open_tasks.get(user)
        if waiting:
            tid = waiting[0]
            self._resolve(user, tid)
            return self.tasks[tid]
        tid = self.last_task_id.get(user)
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
        self.pending_skip = None

    def _on_no_events(self, m):
        """The collector gave up on this client: no skip or S3 line follows."""
        waiting = self.open_tasks.get(m.group(2))
        if waiting:
            self._resolve(m.group(2), waiting[0])

    def _on_status(self, m):
        """A status line means the collector is done with that task."""
        task = self.tasks.get(m.group(3))
        if task is None:
            return
        self._resolve(task.user, task.task_id)
        attr = STATUS_TIMES.get(m.group(4))
        if attr:
            setattr(task, attr, parse_ts(m.group(1)))


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
