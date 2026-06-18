#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys

import diskcache

# Add the parent directory of this script (src) to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


import traceback
import contextlib

import umrs.server.config as _server_config
from umrs.admin.daemon_signal import (
    consume_reload_request,
    install_reload_signal_handler,
    remove_pid_file,
    write_pid_file,
)
from umrs.core.MessageQueue import MessageQueue
from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.TaskManager import TaskManager
from umrs.core.thread_pool import PoolConfig, ServerOverloadedError, ThreadPoolManager
from umrs.inference.embedding_backend import LocalEmbeddingBackend
from umrs.server.config import ENABLE_PRETTY_PRINT, get_default_db_subdir
from umrs.server.durable_job_runtime import start_durable_job_runtime_from_env
from umrs.server.commands import HandlerContext, build_registry
from umrs.server.job_tracking_handler import build_durable_job_backend_from_env
from umrs.server.json_config import load_server_cfg_file
from umrs.server.maintenance import (
    check_duplicated_database,
    check_objects,
    filter_outcome,
    recompute_all_objects,
    reload_configuration,
    update_objects,
)
from umrs.server.model_registry_handler import (
    build_durable_model_backend_from_env,
)
from umrs.server.sql import create_umrs_server_db_user, create_server_db_pool
from umrs.server.provisioning import (
    ServerCfg,
    configure_user,
    create_s3_sql_users,
    is_client_configured,
    reconcile_status_on_startup,
)
from umrs.shared.config import (
    get_default_json_server_file,
    get_default_tmp_subdir,
    get_default_umrs_cfg_dir,
    get_umrs_cfg_env_var_name,
)
from umrs.shared.pretty_print import pretty_print_cfg

_status_store_global = None
logger = logging.getLogger(__name__)


def configure_server_logging():
    """Route Python logger output into the server log stream."""
    level_name = os.getenv("UMRS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# --------------------------- Boot and hot-reload ----------------------------

def _init_session_curve_provider():
    """Create and register the shared session CURVE provider when enabled.

    No-op unless ``UMRS_ZMQ_AUTH_TYPE=session``. When session mode is active
    this installs a single process-wide :class:`SessionCredentialsProvider`
    (started, so its TTL sweep thread runs) that is shared by the ZMQ ZAP
    authenticators and the HTTP ``/umrs/zmq/session`` API. Idempotent.
    """
    try:
        from umrs.core.zmq_security import ZMQSecurityConfig

        if not ZMQSecurityConfig.from_env().is_session_mode:
            return

        from umrs.core.session_curve_provider import (
            SessionCredentialsProvider,
            get_shared_session_provider,
            set_shared_session_provider,
        )

        if get_shared_session_provider() is not None:
            return

        provider = SessionCredentialsProvider()
        provider.start()
        set_shared_session_provider(provider)
        logger.info("Session CURVE provider initialized (UMRS_ZMQ_AUTH_TYPE=session)")
    except Exception:
        logger.exception("Failed to initialize session CURVE provider")
        raise


def perform_initial_config(configure_file, status_store):
    if not os.path.exists(configure_file):
        sys.exit(f"Configuration file '{configure_file}' not found.")

    admin_s3_configuration_file, users, server_cfg = load_server_cfg_file(configure_file)
    check_duplicated_database(users)
    print("Configuration loaded successfully.")

    # Get the directory containing configure_file
    umrs_dir = os.path.dirname(configure_file)

    # Define the temp directory path inside that directory
    tmp_dir = os.path.join(umrs_dir, get_default_tmp_subdir())

    # Create the tmp_dir if it does not exist
    os.makedirs(tmp_dir, exist_ok=True)

    # Session-CURVE: when UMRS_ZMQ_AUTH_TYPE=session, create the process-wide
    # SessionCredentialsProvider BEFORE binding ZMQ sockets so both the ZAP
    # authenticator (control/data sockets) and the HTTP session API share one
    # in-memory registry. No-op in any other auth mode.
    _init_session_curve_provider()

    control_mq = MessageQueue(role='server', queue_type='control', timeout_ms=1000, tmp_dir = tmp_dir, status_store=status_store)
    # Align MessageQueue debug with server pretty-print setting (off by default)
    try:
        control_mq.set_debug(ENABLE_PRETTY_PRINT)
    except Exception as e:
        # Debug flag is optional; log but do not raise
        print(f"Warning: failed to set debug mode on control queue: {e}")
    print ("Subscribed to zeroMQ message queue")

    for user in users:
        create_s3_sql_users(user, admin_s3_configuration_file)

    server_cfg = ServerCfg()
    for user in users:
        configure_user(user, server_cfg)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-->  Models will be run on device: {device}")

    embedding_backend = LocalEmbeddingBackend()
    print("\n######################\nServer configuration done!\n######################\n")

    return control_mq, server_cfg, device, embedding_backend



#########################        Main server loop        #########################
##################################################################################

def make_update_status_callback(task_manager, task_id):
    """
    Return a callback function that updates the task status for the given task_id.
    The returned function accepts a `status` argument when called.
    """
    def callback(status):
        task_manager.tasks_status.update_status(task_id, status)
    return callback

def server_event_loop(
    task_manager,
    control_mq,
    server_cfg,
    device,
    embedding_backend,
    configure_file,
    job_command_backend=None,
    model_command_backend=None,
):
    buckets_cfg = server_cfg.buckets_cfg
    user_config_map = server_cfg.user_config_map
    status_store = control_mq.get_status_store()
    assert status_store is not None
    assert status_store == task_manager.active_threads.get_status_store()

    ctx = HandlerContext(
        task_manager=task_manager,
        control_mq=control_mq,
        server_cfg=server_cfg,
        device=device,
        embedding_backend=embedding_backend,
        configure_file=configure_file,
        job_command_backend=job_command_backend,
        model_command_backend=model_command_backend,
    )
    registry = build_registry()

    try:
        while True:
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######", flush=True)
            task_manager.tasks_status.pretty_print_tasks(limit=20)

            # Honor SIGHUP-triggered hot reload from the offline admin CLI.
            if consume_reload_request():
                try:
                    outcome = reload_configuration(
                        configure_file, buckets_cfg, user_config_map,
                        log_output=ENABLE_PRETTY_PRINT,
                    )
                    filter_outcome(task_manager.active_threads, outcome, buckets_cfg, "reload")
                    if "error" in outcome:
                        print(f"Warning: SIGHUP reload reported errors: {outcome['error']}")
                    else:
                        print("SIGHUP reload complete.")
                except Exception as exc:  # noqa: BLE001 - best-effort reload
                    print(f"ERROR: SIGHUP reload raised: {exc}", flush=True)

            # Log pool metrics once per iteration
            try:
                metrics = task_manager.active_threads.pool.metrics()
                logger.debug(
                    "Pool: active=%d pending=%d (max %d/%d)",
                    metrics.active_count, metrics.pending_count,
                    metrics.max_workers, metrics.max_pending,
                )
            except Exception:
                pass

            try:
                result = control_mq.server_control_queue_loop()
            except Exception as e:
                print(f"ERROR in control queue loop: {e}", flush=True)
                traceback.print_exc()
                continue
            if not result:
                continue
            cmd, payload = result

            task_id = task_manager.tasks_status.create_task(cmd, payload)
            update_status_func = make_update_status_callback(task_manager, task_id)

            registry.dispatch(cmd, payload, update_status_func, ctx, task_id)

            finished_threads_ident_list = task_manager.active_threads.clean_finished_threads()
            logger.debug("finished_threads=%s", finished_threads_ident_list)
            task_manager.update_status_by_threads_ident_list(finished_threads_ident_list, TaskStatus.COMPLETED)

            # After any event and cleanup, update per-user task status summaries
            try:
                users = {entry.s3_username for entry in buckets_cfg.values()}
                users = {u for u in users if u}
                task_manager.active_threads.record_status_for_users_by_key(users, TaskCmd.PROCESS_OBJECTS)
            except Exception as e:
                # Non-fatal: do not disrupt server loop on status recording errors
                print(f"Warning: failed to update task status: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Waiting for running tasks to finish...")
        finished_threads_ident_list = task_manager.active_threads.waitall()
        task_manager.update_status_by_threads_ident_list(finished_threads_ident_list, TaskStatus.COMPLETED)
        task_manager.active_threads.close()
        task_manager.tasks_status.close()
        print("All tasks finished. Exiting now.")
        control_mq.close()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL: Unhandled exception in server event loop: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            task_manager.active_threads.close()
            task_manager.tasks_status.close()
            control_mq.close()
        except Exception as e:
            print(f"Warning: error during shutdown cleanup: {e}")
        sys.exit(1)

def handle_maintenance_flags(task_manager, args, buckets_cfg, device, embedding_backend, configure_file, server_cfg):
    # do in this order
    if args.force_recompute:
        recompute_all_objects(task_manager.active_threads, None, buckets_cfg, device, embedding_backend, TaskCmd.RECOMPUTE, log_output=ENABLE_PRETTY_PRINT)
    if args.force_update:
        update_objects(task_manager.active_threads, None, buckets_cfg, device, embedding_backend, TaskCmd.UPDATE, verbose=False, log_output=ENABLE_PRETTY_PRINT)
    if args.check_objects:
        check_objects(task_manager.active_threads, None, buckets_cfg, TaskCmd.CHECK, verbose=False, log_output=ENABLE_PRETTY_PRINT)
    if args.check_incomplete_command:
        incomplete_commands = task_manager.tasks_status.list_incomplete_tasks(mark_override=True)
        for command, info in incomplete_commands.items():
            client_id, cmd, payload = command
            task_id = int(info["id"])
            status = info["status"]

            print(f"==> Recovering {cmd} for {client_id} with status: {status}")

            # create a callback with task_id embedded
            update_status_func = make_update_status_callback(task_manager, task_id)

            if not is_client_configured(client_id, server_cfg):
                print(f"Client {client_id} is not configured. Skipping recovery.")
                update_status_func(TaskStatus.ERROR)
                continue

            if cmd == TaskCmd.PROCESS_OBJECTS:
                s3_objects = json.loads(payload)
                thread_ident = task_manager.active_threads.launch_s3_threads(s3_objects,
                                                                             buckets_cfg, device, embedding_backend,
                                                                             cmd,
                                                                             update_status_callback=update_status_func)
                task_manager.associate_thread_with_task(thread_ident, task_id)
            elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK):
                user_config_map = server_cfg.user_config_map
                thread_ident = _handle_maintenance_command(task_manager.active_threads, cmd, client_id, buckets_cfg,
                                                            device, embedding_backend, configure_file, user_config_map, cmd,
                                                            update_status_callback=update_status_func)
                if thread_ident is None:
                    update_status_func(TaskStatus.COMPLETED)
                else:
                    task_manager.associate_thread_with_task(thread_ident, task_id)

            elif cmd == TaskCmd.FILE:
                user_config_map = server_cfg.user_config_map
                files = json.loads(payload)
                for file in files:
                    if not os.path.exists(file):
                        print(f"Trying to recover file: {file}, but it no longer exists. Skipping.")
                        continue
                    print(f"Recovering file: {file}")

                    ok, uname, dest = _persist_uploaded_user_config(
                        file, configure_file, user_config_map, expected_username=client_id
                    )
                    if ok:
                        print(f"User config persisted for '{uname}' -> {dest}")
                    else:
                        print(f"Note: file '{file}' was not recognized as a valid user config or there was an identity mismatch; left in place.")

                update_status_func(TaskStatus.COMPLETED)
            else:
                print("\tNothing to do for command of type {cmd}")

def check_umrs_config_directory():
    umrs_config_dir = os.getenv(get_umrs_cfg_env_var_name())

    if umrs_config_dir:
        if os.path.isdir(umrs_config_dir):
            return umrs_config_dir
        else:
            return None
    else:
        default_dir = get_default_umrs_cfg_dir()
        if os.path.isdir(default_dir):
            return default_dir
        else:
            return None
# ---------------------------------- CLI ------------------------------------

def main():
    # Offline admin CLI: route `umrs-server admin ...` before any daemon
    # bootstrap so it works without a configured UMRS_CONFIG directory.
    if len(sys.argv) > 1 and sys.argv[1] == "admin":
        from umrs.cli.admin import main as admin_main
        return admin_main(sys.argv[2:])

    configure_server_logging()

    # check if we have a correct configuration directory
    umrs_cfg_dir = check_umrs_config_directory()
    if umrs_cfg_dir:
        default_json_server_path = f"{umrs_cfg_dir}/{get_default_json_server_file()}"
    else:
        sys.exit(f"No valid UMRS configuration directory. The default directory is {get_default_umrs_cfg_dir()} or the value of the environment variable {get_umrs_cfg_env_var_name()}. Check it exists.")

    parser = argparse.ArgumentParser(description="UMRS server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--conf', required=(default_json_server_path is None), default=default_json_server_path, help=f"Config file path. The default directory is {get_default_umrs_cfg_dir()} or the value of the environment variable {get_umrs_cfg_env_var_name()}.")
    parser.add_argument('--force-recompute', action='store_true', help='Recompute all embeddings by scanning all S3 directories. Existing entries will be fully overwritten.')
    parser.add_argument('--force-update', action='store_true', help='Recompute embeddings only for missing or mismatched entries. Only those entries will be updated in the database.')
    parser.add_argument('--check-objects', action='store_true', help='Check all the buckets to see if the corresponding objects are stored in the database')
    parser.add_argument('--pretty-print', action='store_true', help='Enable server-side pretty-printed tables and summaries (default: off)')
    parser.add_argument('--check-incomplete-command', action='store_true', help='Force checking for incomplete commands after a crash (default: off)')
    parser.add_argument('--pool-max-workers', type=int, default=None,
                        help='Max worker threads in the pool (default: UMRS_POOL_MAX_WORKERS or 16)')
    parser.add_argument('--pool-max-pending', type=int, default=None,
                        help='Max pending tasks before rejecting requests (default: UMRS_POOL_MAX_PENDING or 64)')
    args = parser.parse_args()
    configure_file = args.conf

    global ENABLE_PRETTY_PRINT
    ENABLE_PRETTY_PRINT = args.pretty_print
    _server_config.ENABLE_PRETTY_PRINT = args.pretty_print

    print(f"Starting UMRS server with configuration file: {configure_file}")
    status_store = diskcache.Cache(f"{get_default_tmp_subdir()}/status_store", size_limit = 500*1024*1024)

    print("Initialized status store")
    thread_dict = diskcache.Cache(f"{get_default_tmp_subdir()}/thread_dict", size_limit = 500*1024*1024)

    print("Initialized thread dictionary")

    # Build thread pool with optional CLI overrides
    pool_kwargs = {}
    if args.pool_max_workers is not None:
        pool_kwargs["max_workers"] = args.pool_max_workers
    if args.pool_max_pending is not None:
        pool_kwargs["max_pending"] = args.pool_max_pending
    thread_pool = ThreadPoolManager(PoolConfig(**pool_kwargs)) if pool_kwargs else ThreadPoolManager()

    create_umrs_server_db_user()
    task_db_pool = create_server_db_pool()
    task_manager = TaskManager(status_store, thread_dict, task_db_pool, thread_pool=thread_pool, reset=False)

    print("Performing initial configuration...")
    control_mq, server_cfg, device, embedding_backend = perform_initial_config(configure_file, status_store)

    print("Initial configuration done.")
    reconcile_status_on_startup(task_manager.active_threads, server_cfg.buckets_cfg)

    durable_job_runtime = start_durable_job_runtime_from_env()
    if durable_job_runtime is not None:
        job_command_backend = durable_job_runtime.command_backend
        print("Job commands using durable PostgreSQL backend with active scheduler.")
    else:
        job_command_backend = build_durable_job_backend_from_env()

    if job_command_backend is None:
        print("Job commands unavailable: durable PostgreSQL backend could not start.")
    elif durable_job_runtime is None:
        print("Job commands using durable PostgreSQL backend.")

    model_command_backend = build_durable_model_backend_from_env()
    if model_command_backend is None:
        print("Model commands using legacy JSON catalog fallback.")
    else:
        print("Model commands using durable PostgreSQL backend.")

    # Start HTTP sidecar so the Platform Gateway can proxy REST requests
    # back to this UMRS instance (port 8082 by default). The sidecar runs
    # in a daemon thread and respects UMRS_HTTP_SIDECAR_ENABLED env var.
    try:
        from umrs.server.http_sidecar import start_http_sidecar

        _sidecar_thread = start_http_sidecar()
        if _sidecar_thread is not None:
            print(f"HTTP sidecar started (thread={_sidecar_thread.name}).")
        else:
            print("HTTP sidecar disabled by configuration.")
    except Exception as _sc_exc:
        print(f"Warning: HTTP sidecar failed to start: {_sc_exc}")

    handle_maintenance_flags(task_manager, args, server_cfg.buckets_cfg, device, embedding_backend, configure_file, server_cfg)

    print("Entering main server event loop.")
    cfg_summary = pretty_print_cfg(server_cfg, print_output=ENABLE_PRETTY_PRINT)

    print(f"Server configuration summary:\n{cfg_summary}")
    task_manager.active_threads.record_status("system", "config", {"summary": cfg_summary})

    print("Server is ready to process tasks.")
    # Install SIGHUP handler and write a PID file so the offline admin CLI
    # can signal this process to hot-reload user records after mutations.
    try:
        install_reload_signal_handler()
    except (ValueError, OSError) as exc:
        # ValueError is raised if we are not on the main thread; OSError
        # covers platforms without SIGHUP. Admin-driven hot reload will be
        # unavailable in that case, which is acceptable.
        print(f"Warning: could not install SIGHUP handler: {exc}")
    pid_file_path = None
    try:
        pid_file_path = write_pid_file()
        print(f"PID file written: {pid_file_path}")
    except OSError as exc:
        print(f"Warning: could not write PID file: {exc}")
    try:
        server_event_loop(
            task_manager,
            control_mq,
            server_cfg,
            device,
            embedding_backend,
            configure_file,
            job_command_backend=job_command_backend,
            model_command_backend=model_command_backend,
        )
    finally:
        if pid_file_path is not None:
            with contextlib.suppress(OSError):
                remove_pid_file(pid_file_path)
        if durable_job_runtime is not None:
            try:
                durable_job_runtime.shutdown()
            except Exception:
                logger.exception("Failed to shut down durable job runtime")
        elif job_command_backend is not None:
            try:
                job_command_backend.close()
            except Exception:
                logger.exception("Failed to close durable job command backend")
        if model_command_backend is not None:
            try:
                model_command_backend.close()
            except Exception:
                logger.exception("Failed to close durable model command backend")


if __name__ == "__main__":
    main()
