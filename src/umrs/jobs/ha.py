"""
HA Coordination for UMRS instances (U-703, U-715).

This module provides distributed locking and heartbeat mechanisms
for running multiple UMRS instances in a high-availability cluster.

Key Classes:
  - HeartbeatManager: Updates instance heartbeats, detects dead instances
  - DistributedLockManager: Advisory locks for exclusive resource access
  - ClusterCoordinator: High-level cluster coordination
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# Default intervals
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_LOCK_TTL_SECONDS = 60
DEFAULT_DEAD_THRESHOLD_SECONDS = 90


class HeartbeatError(Exception):
    """Raised when heartbeat operation fails."""

    pass


class LockError(Exception):
    """Base exception for lock errors."""

    pass


class LockAcquisitionError(LockError):
    """Raised when lock cannot be acquired."""

    pass


class LockNotHeldError(LockError):
    """Raised when trying to release a lock we don't hold."""

    pass


class InstanceInfo:
    """Information about a UMRS instance."""

    def __init__(
        self,
        instance_id: str,
        hostname: str,
        pid: int,
        started_at: datetime,
        last_heartbeat_at: datetime,
        status: str = "active",
        capabilities: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.instance_id = instance_id
        self.hostname = hostname
        self.pid = pid
        self.started_at = started_at
        self.last_heartbeat_at = last_heartbeat_at
        self.status = status
        self.capabilities = capabilities or {}
        self.metadata = metadata or {}

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.last_heartbeat_at).total_seconds()


class HeartbeatManager:
    """
    Manages UMRS instance heartbeats for HA coordination (U-715).

    Each UMRS instance:
    1. Registers itself on startup
    2. Sends periodic heartbeats
    3. Detects dead instances (missed heartbeats)
    4. Cleans up resources from dead instances
    """

    def __init__(
        self,
        db_pool,
        instance_id: str | None = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        dead_threshold: float = DEFAULT_DEAD_THRESHOLD_SECONDS,
    ):
        self.db_pool = db_pool
        self.instance_id = instance_id or self._generate_instance_id()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.heartbeat_interval = heartbeat_interval
        self.dead_threshold = dead_threshold

        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._registered = False
        self._capabilities: dict[str, Any] = {}

        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    def _generate_instance_id(self) -> str:
        """Generate unique instance ID."""
        return f"umrs-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    def register(self, capabilities: dict[str, Any] | None = None) -> None:
        """
        Register this instance in the cluster.

        Args:
            capabilities: What this instance can do (scheduler, monitor, etc.)
        """
        self._capabilities = capabilities or {"roles": ["all"]}

        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO umrs_instances (
                        instance_id, hostname, pid, started_at,
                        last_heartbeat_at, status, capabilities
                    )
                    VALUES (%s, %s, %s, NOW(), NOW(), 'active', %s)
                    ON CONFLICT (instance_id) DO UPDATE SET
                        hostname = EXCLUDED.hostname,
                        pid = EXCLUDED.pid,
                        last_heartbeat_at = NOW(),
                        status = 'active',
                        capabilities = EXCLUDED.capabilities
                """,
                    (
                        self.instance_id,
                        self.hostname,
                        self.pid,
                        json.dumps(self._capabilities) if self._capabilities else "{}",
                    ),
                )
            conn.commit()

        self._registered = True
        logger.info(f"Registered UMRS instance: {self.instance_id}")

    def start_heartbeat_loop(self) -> None:
        """Start background heartbeat thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            logger.warning("Heartbeat loop already running")
            return

        if not self._registered:
            self.register()

        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"umrs-heartbeat-{self.instance_id[:8]}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        logger.info(f"Started heartbeat loop (interval={self.heartbeat_interval}s)")

    def stop_heartbeat_loop(self) -> None:
        """Stop background heartbeat thread."""
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None
        logger.info("Stopped heartbeat loop")

    def _heartbeat_loop(self) -> None:
        """Background thread that sends periodic heartbeats."""
        while not self._stop_event.is_set():
            try:
                self.send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

            self._stop_event.wait(self.heartbeat_interval)

    def send_heartbeat(self) -> None:
        """Send heartbeat to update last_heartbeat_at."""

        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE umrs_instances
                    SET last_heartbeat_at = NOW()
                    WHERE instance_id = %s
                """,
                    (self.instance_id,),
                )

                if cur.rowcount == 0:
                    # Re-register if our row was deleted
                    logger.warning("Instance row missing, re-registering")
                    self._registered = False
                    self.register(self._capabilities)
            conn.commit()

    def list_instances(self, include_dead: bool = False) -> list[InstanceInfo]:
        """
        List all known UMRS instances.

        Args:
            include_dead: Include instances that missed heartbeats
        """
        with self.db_pool.get_conn() as conn, conn.cursor() as cur:
            if include_dead:
                cur.execute("""
                        SELECT instance_id, hostname, pid, started_at,
                               last_heartbeat_at, status, capabilities, metadata
                        FROM umrs_instances
                        ORDER BY started_at DESC
                    """)
            else:
                cur.execute(
                    """
                        SELECT instance_id, hostname, pid, started_at,
                               last_heartbeat_at, status, capabilities, metadata
                        FROM umrs_instances
                        WHERE status = 'active'
                          AND last_heartbeat_at > NOW() - INTERVAL '%s seconds'
                        ORDER BY started_at DESC
                    """,
                    (self.dead_threshold,),
                )

            rows = cur.fetchall()

        return [
            InstanceInfo(
                instance_id=row[0],
                hostname=row[1],
                pid=row[2],
                started_at=row[3],
                last_heartbeat_at=row[4],
                status=row[5],
                capabilities=row[6] or {},
                metadata=row[7] or {},
            )
            for row in rows
        ]

    def find_dead_instances(self) -> list[InstanceInfo]:
        """Find instances that have missed heartbeats."""
        with self.db_pool.get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT instance_id, hostname, pid, started_at,
                           last_heartbeat_at, status, capabilities, metadata
                    FROM umrs_instances
                    WHERE status = 'active'
                      AND last_heartbeat_at < NOW() - INTERVAL '%s seconds'
                """,
                (self.dead_threshold,),
            )

            rows = cur.fetchall()

        return [
            InstanceInfo(
                instance_id=row[0],
                hostname=row[1],
                pid=row[2],
                started_at=row[3],
                last_heartbeat_at=row[4],
                status=row[5],
                capabilities=row[6] or {},
                metadata=row[7] or {},
            )
            for row in rows
        ]

    def mark_instance_dead(self, instance_id: str) -> None:
        """Mark an instance as dead (for cleanup)."""
        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE umrs_instances
                    SET status = 'dead'
                    WHERE instance_id = %s
                """,
                    (instance_id,),
                )
            conn.commit()

        logger.info(f"Marked instance {instance_id} as dead")

    def deregister(self) -> None:
        """Deregister this instance (graceful shutdown)."""
        self.stop_heartbeat_loop()

        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE umrs_instances
                    SET status = 'draining'
                    WHERE instance_id = %s
                """,
                    (self.instance_id,),
                )
            conn.commit()

        self._registered = False
        logger.info(f"Deregistered UMRS instance: {self.instance_id}")

    def _cleanup_on_exit(self) -> None:
        """Atexit handler to clean up."""
        try:
            if self._registered:
                self.deregister()
        except Exception:
            pass  # Best effort cleanup


class DistributedLockManager:
    """
    Distributed lock manager for HA coordination (U-703).

    Uses PostgreSQL table-based locks with TTL for automatic cleanup.
    Supports different lock types:
      - scheduler: Exclusive scheduler leadership
      - job: Per-job exclusive access
      - resource: Resource-level locking (e.g., GPU node)
    """

    def __init__(
        self,
        db_pool,
        instance_id: str,
        default_ttl: float = DEFAULT_LOCK_TTL_SECONDS,
    ):
        self.db_pool = db_pool
        self.instance_id = instance_id
        self.hostname = socket.gethostname()
        self.default_ttl = default_ttl
        self._held_locks: dict[str, datetime] = {}  # lock_id -> acquired_at

    def _generate_lock_id(self, lock_type: str, resource_id: str) -> str:
        """Generate deterministic lock ID."""
        return f"{lock_type}:{resource_id}"

    def acquire(
        self,
        lock_type: str,
        resource_id: str,
        ttl: float | None = None,
        wait: bool = False,
        wait_timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Acquire a distributed lock.

        Args:
            lock_type: Lock category (scheduler, job, resource)
            resource_id: What to lock (job_id, node_name, etc.)
            ttl: Time-to-live in seconds (auto-release after)
            wait: If True, wait for lock availability
            wait_timeout: Max seconds to wait
            metadata: Additional lock metadata

        Returns:
            True if lock acquired, False otherwise
        """
        import json

        lock_id = self._generate_lock_id(lock_type, resource_id)
        ttl_seconds = ttl or self.default_ttl

        start_time = time.time()

        while True:
            # First, clean up expired locks
            self._cleanup_expired()

            try:
                with self.db_pool.get_conn() as conn:
                    with conn.cursor() as cur:
                        # Try to insert new lock
                        cur.execute(
                            """
                            INSERT INTO job_locks (
                                lock_id, lock_type, resource_id,
                                owner_instance_id, owner_hostname,
                                acquired_at, expires_at, heartbeat_at, metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '%s seconds', NOW(), %s)
                            ON CONFLICT (lock_type, resource_id) DO NOTHING
                            RETURNING lock_id
                        """,
                            (
                                lock_id,
                                lock_type,
                                resource_id,
                                self.instance_id,
                                self.hostname,
                                ttl_seconds,
                                json.dumps(metadata or {}),
                            ),
                        )

                        result = cur.fetchone()
                        if result:
                            conn.commit()
                            self._held_locks[lock_id] = datetime.utcnow()
                            logger.debug(f"Acquired lock: {lock_id}")
                            return True

                        # Check if we already own it
                        cur.execute(
                            """
                            SELECT owner_instance_id, expires_at
                            FROM job_locks
                            WHERE lock_id = %s
                        """,
                            (lock_id,),
                        )
                        existing = cur.fetchone()

                        if existing and existing[0] == self.instance_id:
                            # We already own it, refresh TTL
                            cur.execute(
                                """
                                UPDATE job_locks
                                SET expires_at = NOW() + INTERVAL '%s seconds',
                                    heartbeat_at = NOW()
                                WHERE lock_id = %s
                            """,
                                (ttl_seconds, lock_id),
                            )
                            conn.commit()
                            logger.debug(f"Refreshed existing lock: {lock_id}")
                            return True

                    conn.rollback()

            except Exception as e:
                logger.error(f"Lock acquisition error: {e}")

            # If not waiting or timeout exceeded, return False
            if not wait:
                return False

            elapsed = time.time() - start_time
            if elapsed >= wait_timeout:
                logger.warning(f"Lock acquisition timeout: {lock_id}")
                return False

            # Sleep briefly and retry
            time.sleep(0.5)

    def release(self, lock_type: str, resource_id: str) -> bool:
        """
        Release a distributed lock.

        Args:
            lock_type: Lock category
            resource_id: What was locked

        Returns:
            True if lock released, False if not held
        """
        lock_id = self._generate_lock_id(lock_type, resource_id)

        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM job_locks
                    WHERE lock_id = %s AND owner_instance_id = %s
                    RETURNING lock_id
                """,
                    (lock_id, self.instance_id),
                )

                result = cur.fetchone()
            conn.commit()

        if result:
            self._held_locks.pop(lock_id, None)
            logger.debug(f"Released lock: {lock_id}")
            return True

        return False

    def refresh(
        self, lock_type: str, resource_id: str, ttl: float | None = None
    ) -> bool:
        """
        Refresh lock TTL (extend expiration).

        Args:
            lock_type: Lock category
            resource_id: What is locked
            ttl: New TTL (default: use default_ttl)

        Returns:
            True if refreshed, False if not held
        """
        lock_id = self._generate_lock_id(lock_type, resource_id)
        ttl_seconds = ttl or self.default_ttl

        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE job_locks
                    SET expires_at = NOW() + INTERVAL '%s seconds',
                        heartbeat_at = NOW()
                    WHERE lock_id = %s AND owner_instance_id = %s
                    RETURNING lock_id
                """,
                    (ttl_seconds, lock_id, self.instance_id),
                )

                result = cur.fetchone()
            conn.commit()

        if result:
            logger.debug(f"Refreshed lock: {lock_id}")
            return True

        return False

    @contextmanager
    def lock(
        self,
        lock_type: str,
        resource_id: str,
        ttl: float | None = None,
        wait: bool = True,
        wait_timeout: float = 30.0,
    ):
        """
        Context manager for acquiring/releasing lock.

        Usage:
            with lock_manager.lock("job", job_id):
                # exclusive access to job
                ...

        Raises:
            LockAcquisitionError: If lock cannot be acquired
        """
        acquired = self.acquire(
            lock_type=lock_type,
            resource_id=resource_id,
            ttl=ttl,
            wait=wait,
            wait_timeout=wait_timeout,
        )

        if not acquired:
            raise LockAcquisitionError(
                f"Failed to acquire lock: {lock_type}:{resource_id}"
            )

        try:
            yield
        finally:
            self.release(lock_type, resource_id)

    def is_held(self, lock_type: str, resource_id: str) -> bool:
        """Check if we hold a specific lock."""
        lock_id = self._generate_lock_id(lock_type, resource_id)
        return lock_id in self._held_locks

    def get_lock_owner(self, lock_type: str, resource_id: str) -> str | None:
        """Get the instance that holds a lock (if any)."""
        lock_id = self._generate_lock_id(lock_type, resource_id)

        with self.db_pool.get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT owner_instance_id
                    FROM job_locks
                    WHERE lock_id = %s AND expires_at > NOW()
                """,
                (lock_id,),
            )

            result = cur.fetchone()

        return result[0] if result else None

    def release_all(self) -> int:
        """Release all locks held by this instance."""
        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM job_locks
                    WHERE owner_instance_id = %s
                    RETURNING lock_id
                """,
                    (self.instance_id,),
                )

                released = cur.fetchall()
            conn.commit()

        self._held_locks.clear()
        count = len(released)
        if count > 0:
            logger.info(f"Released {count} locks for instance {self.instance_id}")
        return count

    def _cleanup_expired(self) -> int:
        """Clean up expired locks."""
        with self.db_pool.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM job_locks
                    WHERE expires_at < NOW()
                    RETURNING lock_id
                """)

                expired = cur.fetchall()
            conn.commit()

        count = len(expired)
        if count > 0:
            logger.debug(f"Cleaned up {count} expired locks")
        return count


class ClusterCoordinator:
    """
    High-level cluster coordination for UMRS HA.

    Combines HeartbeatManager and DistributedLockManager for:
    - Leader election (scheduler leadership)
    - Dead instance detection and cleanup
    - Exclusive resource access
    """

    def __init__(
        self,
        db_pool,
        instance_id: str | None = None,
        capabilities: dict[str, Any] | None = None,
    ):
        self.db_pool = db_pool

        # Create heartbeat manager (generates instance_id if not provided)
        self.heartbeat = HeartbeatManager(db_pool, instance_id)
        self.instance_id = self.heartbeat.instance_id

        # Create lock manager using same instance_id
        self.locks = DistributedLockManager(db_pool, self.instance_id)

        self._capabilities = capabilities or {}
        self._is_scheduler = False

    def start(self) -> None:
        """Start cluster participation."""
        self.heartbeat.register(self._capabilities)
        self.heartbeat.start_heartbeat_loop()
        logger.info(f"Cluster coordinator started: {self.instance_id}")

    def stop(self) -> None:
        """Stop cluster participation (graceful shutdown)."""
        self._is_scheduler = False
        self.locks.release_all()
        self.heartbeat.deregister()
        logger.info("Cluster coordinator stopped")

    def try_become_scheduler(self, ttl: float = 60.0) -> bool:
        """
        Try to become the scheduler leader.

        Only one instance can be scheduler at a time.
        Must periodically refresh to maintain leadership.

        Returns:
            True if we are now scheduler, False otherwise
        """
        acquired = self.locks.acquire("scheduler", "leader", ttl=ttl, wait=False)
        self._is_scheduler = acquired

        if acquired:
            logger.info("Became scheduler leader")

        return acquired

    def refresh_scheduler_leadership(self, ttl: float = 60.0) -> bool:
        """Refresh scheduler leadership (call periodically)."""
        if not self._is_scheduler:
            return False

        refreshed = self.locks.refresh("scheduler", "leader", ttl=ttl)
        if not refreshed:
            self._is_scheduler = False
            logger.warning("Lost scheduler leadership")

        return refreshed

    def is_scheduler(self) -> bool:
        """Check if we are currently the scheduler."""
        return self._is_scheduler and self.locks.is_held("scheduler", "leader")

    def get_scheduler_instance(self) -> str | None:
        """Get the current scheduler instance ID (if any)."""
        return self.locks.get_lock_owner("scheduler", "leader")

    def cleanup_dead_instances(self) -> int:
        """
        Find and clean up resources from dead instances.

        Returns:
            Number of dead instances cleaned up
        """
        dead_instances = self.heartbeat.find_dead_instances()

        for instance in dead_instances:
            logger.info(f"Cleaning up dead instance: {instance.instance_id}")

            # Release locks held by dead instance
            with self.db_pool.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM job_locks
                        WHERE owner_instance_id = %s
                    """,
                        (instance.instance_id,),
                    )
                conn.commit()

            # Mark instance as dead
            self.heartbeat.mark_instance_dead(instance.instance_id)

        return len(dead_instances)


# Import json at module level for use in methods
import json
