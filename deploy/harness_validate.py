#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umrs.orchestrator.base import EndpointInfo, JobStatus
from umrs.orchestrator.service import OrchestrationService, ProvisionRequest as ServiceProvisionRequest
from umrs.orchestrator.readiness import ReadinessConfig, ReadinessPoller
from umrs.orchestrator.site_config import SiteConfig, load_site_config
from umrs.orchestrator.slurm_podman_backend import SlurmPodmanBackend


@dataclass(frozen=True)
class ValidationCase:
    name: str
    profiles: tuple[str, ...]
    expect_models: bool
    expect_vision: bool


CASE_MATRIX = {
    "text": ValidationCase(
        name="text",
        profiles=("harness", "vllm"),
        expect_models=True,
        expect_vision=False,
    ),
    "vision": ValidationCase(
        name="vision",
        profiles=("harness", "vision-bentoml"),
        expect_models=False,
        expect_vision=True,
    ),
    "full": ValidationCase(
        name="full",
        profiles=("harness", "vllm", "vision-bentoml"),
        expect_models=True,
        expect_vision=True,
    ),
}


def log(message: str) -> None:
    print(f"[validate-harness] {message}")


def profile_string(case: ValidationCase) -> str:
    return " ".join(case.profiles)


def vision_log_candidates() -> tuple[str, ...]:
    return (
        "Loading vision image from Lustre tarball",
        "Loading vision image from shared home tarball",
        "Building vision image on compute node",
    )


async def wait_for_endpoint(
    backend: SlurmPodmanBackend,
    job_id: str,
    timeout_sec: int,
) -> EndpointInfo:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        job = await backend.get_job_status(job_id)
        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"Slurm job {job_id} failed: {job.error_message or 'unknown failure'}")
        if job.status == JobStatus.STOPPED:
            raise RuntimeError(f"Slurm job {job_id} stopped before exposing an endpoint")
        if job.endpoint:
            return job.endpoint
        await asyncio.sleep(5)

    raise TimeoutError(f"Timed out waiting for Slurm job {job_id} to publish a host")


async def wait_for_log_summary(
    log_path: Path,
    expected_profiles: str,
    timeout_sec: int,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if "Launch context:" in text and f"profiles: {expected_profiles}" in text:
                return
        await asyncio.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for retained wrapper log summary in {log_path}"
    )


async def wait_for_log_any(
    log_path: Path,
    candidates: Sequence[str],
    timeout_sec: int,
) -> str:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            for candidate in candidates:
                if candidate in text:
                    return candidate
        await asyncio.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for retained wrapper log evidence in {log_path}: {list(candidates)}"
    )


async def wait_for_models(
    site: SiteConfig,
    endpoint: EndpointInfo,
    timeout_sec: int,
) -> list[str]:
    poller = ReadinessPoller(
        ReadinessConfig(
            timeout_sec=timeout_sec,
            expected_model=site.vllm.model,
            auth_token=site.network.litellm_master_key or None,
        )
    )
    try:
        result = await poller.wait_for_ready(
            job_id=f"models@{endpoint.node}",
            host=endpoint.node,
            port=endpoint.litellm_port,
        )
    finally:
        await poller.close()

    if not result.is_ready:
        raise RuntimeError(
            f"LiteLLM readiness failed on {endpoint.node}:{endpoint.litellm_port}: {result.error}"
        )
    return result.models


async def wait_for_vision_health(endpoint: EndpointInfo, timeout_sec: int) -> None:
    url = f"http://{endpoint.node}:{endpoint.litellm_port}/vision/healthz"
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(3)

    raise TimeoutError(
        f"Timed out waiting for vision health at {url}: {last_error or 'no successful response'}"
    )


async def wait_for_cleanup(
    backend: SlurmPodmanBackend,
    job_id: str,
    timeout_sec: int,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        job = await backend.get_job_status(job_id)
        if job.status in (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.UNKNOWN):
            return
        await asyncio.sleep(3)

    raise TimeoutError(f"Timed out waiting for Slurm cleanup of job {job_id}")


async def cleanup_job(
    backend: SlurmPodmanBackend,
    job_id: str,
    timeout_sec: int,
) -> None:
    await backend.cleanup(job_id)
    try:
        await wait_for_cleanup(backend, job_id, timeout_sec=timeout_sec)
        return
    except TimeoutError:
        log(f"Graceful cleanup timed out for Slurm job {job_id}; escalating to KILL")

    await backend.cleanup(job_id, force=True)
    await wait_for_cleanup(backend, job_id, timeout_sec=timeout_sec)


async def cleanup_registered_endpoint(
    service: OrchestrationService,
    job_id: str,
    timeout_sec: int,
) -> None:
    endpoint = service.registry.get_by_job_id(job_id)
    await cleanup_job(service.backend, job_id, timeout_sec=timeout_sec)
    if endpoint is not None:
        service.port_manager.release(endpoint.port, endpoint.host)
        allocated_host = endpoint.metadata.get("allocated_host")
        if allocated_host and allocated_host != endpoint.host:
            service.port_manager.release(endpoint.port, allocated_host)
    service.registry.remove(job_id)


async def run_lifecycle_case(
    site: SiteConfig,
    case: ValidationCase,
    assignment_timeout_sec: int,
    endpoint_timeout_sec: int,
    cleanup_timeout_sec: int,
) -> None:
    if not case.expect_models:
        raise ValueError(
            f"Lifecycle validation requires a case with LiteLLM readiness; got {case.name}"
        )

    profiles_value = profile_string(case)
    readiness_poller = ReadinessPoller(
        ReadinessConfig(
            timeout_sec=endpoint_timeout_sec,
            expected_model=site.vllm.model,
            auth_token=site.network.litellm_master_key or None,
        )
    )
    service = OrchestrationService(readiness_poller=readiness_poller)
    first_job_id: str | None = None
    second_job_id: str | None = None
    third_job_id: str | None = None
    project_id = f"harness-lifecycle-{case.name}-{int(time.time())}"

    async def provision_once(expect_reuse: bool) -> tuple[str, EndpointInfo, Path]:
        request = ServiceProvisionRequest(
            project_id=project_id,
            model=site.vllm.model,
            model_path="",
            ttl_sec=max(600, cleanup_timeout_sec * 3),
            prefer_warm=True,
            min_ttl_for_reuse_sec=60,
            env_vars={"RIVAULT_PROFILES": profiles_value},
            metadata={"readiness_probe": "models"},
        )
        result = await service.provision(request)
        if not result.success or result.endpoint is None or result.job_id is None:
            raise RuntimeError(
                f"Lifecycle provision failed for {case.name}: {result.error or 'no endpoint'}"
            )
        if result.reused != expect_reuse:
            raise RuntimeError(
                f"Lifecycle provision reuse mismatch for {case.name}: "
                f"expected reused={expect_reuse}, got {result.reused}"
            )

        endpoint = EndpointInfo(
            node=result.endpoint.host,
            litellm_port=result.endpoint.port,
        )
        log_path = service.backend._output_dir / f"slurm-{result.job_id}.out"
        return result.job_id, endpoint, log_path

    try:
        log(f"Running lifecycle validation for {case.name} with profiles: {profiles_value}")

        first_job_id, first_endpoint, first_log_path = await provision_once(expect_reuse=False)
        await wait_for_log_summary(first_log_path, profiles_value, timeout_sec=assignment_timeout_sec)
        log(
            f"Lifecycle initial provision ready via Slurm job {first_job_id} at "
            f"{first_endpoint.node}:{first_endpoint.litellm_port}"
        )

        if case.expect_vision:
            marker = await wait_for_log_any(
                first_log_path,
                vision_log_candidates(),
                timeout_sec=assignment_timeout_sec,
            )
            await wait_for_vision_health(first_endpoint, timeout_sec=endpoint_timeout_sec)
            log(f"Lifecycle initial vision image path evidence: {marker}")

        second_job_id, second_endpoint, _second_log_path = await provision_once(expect_reuse=True)
        if second_job_id != first_job_id:
            raise RuntimeError(
                f"Lifecycle reuse expected job {first_job_id}, got {second_job_id}"
            )
        log(
            f"Lifecycle reuse confirmed for {case.name}: reused Slurm job {second_job_id} "
            f"at {second_endpoint.node}:{second_endpoint.litellm_port}"
        )

        log(f"Lifecycle cleanup for reused Slurm job {first_job_id}")
        await cleanup_registered_endpoint(service, first_job_id, timeout_sec=cleanup_timeout_sec)
        if service.registry.get_by_job_id(first_job_id) is not None:
            raise RuntimeError(f"Lifecycle cleanup did not remove registry entry {first_job_id}")
        if not first_log_path.exists():
            raise RuntimeError(f"Lifecycle cleanup removed retained log {first_log_path}")
        log(f"Lifecycle cleanup confirmed for Slurm job {first_job_id}")

        third_job_id, third_endpoint, third_log_path = await provision_once(expect_reuse=False)
        if third_job_id == first_job_id:
            raise RuntimeError(
                f"Lifecycle restart expected a fresh backend after cleanup, but reused {third_job_id}"
            )
        await wait_for_log_summary(third_log_path, profiles_value, timeout_sec=assignment_timeout_sec)
        log(
            f"Lifecycle fresh restart confirmed for {case.name}: new Slurm job {third_job_id} "
            f"at {third_endpoint.node}:{third_endpoint.litellm_port}"
        )

        if case.expect_vision:
            marker = await wait_for_log_any(
                third_log_path,
                vision_log_candidates(),
                timeout_sec=assignment_timeout_sec,
            )
            await wait_for_vision_health(third_endpoint, timeout_sec=endpoint_timeout_sec)
            log(f"Lifecycle restart vision image path evidence: {marker}")

    finally:
        for job_id in (third_job_id, second_job_id, first_job_id):
            if job_id and service.registry.get_by_job_id(job_id) is not None:
                try:
                    await cleanup_registered_endpoint(
                        service,
                        job_id,
                        timeout_sec=cleanup_timeout_sec,
                    )
                except Exception as exc:
                    log(f"Lifecycle cleanup warning for {job_id}: {exc}")
        await service.close()


async def run_case(
    backend: SlurmPodmanBackend,
    site: SiteConfig,
    case: ValidationCase,
    assignment_timeout_sec: int,
    endpoint_timeout_sec: int,
    cleanup_timeout_sec: int,
    skip_cleanup: bool,
) -> None:
    profiles_value = profile_string(case)
    log(f"Submitting {case.name} case with profiles: {profiles_value}")
    job = await backend.submit_job(env={"RIVAULT_PROFILES": profiles_value})
    log_path = backend._output_dir / f"slurm-{job.job_id}.out"

    try:
        await wait_for_log_summary(log_path, profiles_value, timeout_sec=assignment_timeout_sec)
        endpoint = await wait_for_endpoint(backend, job.job_id, timeout_sec=assignment_timeout_sec)
        log(f"Case {case.name} assigned to {endpoint.node}:{endpoint.litellm_port} via Slurm job {job.job_id}")

        if case.expect_models:
            models = await wait_for_models(site, endpoint, timeout_sec=endpoint_timeout_sec)
            log(f"Case {case.name} models ready: {models}")

        if case.expect_vision:
            marker = await wait_for_log_any(
                log_path,
                vision_log_candidates(),
                timeout_sec=assignment_timeout_sec,
            )
            await wait_for_vision_health(endpoint, timeout_sec=endpoint_timeout_sec)
            log(f"Case {case.name} vision image path evidence: {marker}")
            log(f"Case {case.name} vision health ready at /vision/healthz")

    finally:
        if not skip_cleanup:
            log(f"Cleaning up Slurm job {job.job_id}")
            await cleanup_job(backend, job.job_id, timeout_sec=cleanup_timeout_sec)
            log(f"Cleanup confirmed for Slurm job {job.job_id}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the UMRS Slurm harness against supported profile sets."
    )
    parser.add_argument(
        "--site-dir",
        required=True,
        help="Path to deploy/<site> containing site.yaml and serve-model.slurm.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=sorted(CASE_MATRIX),
        help="Validation case to run. Repeat to run multiple cases. Defaults to all.",
    )
    parser.add_argument(
        "--assignment-timeout",
        type=int,
        default=900,
        help="Seconds to wait for Slurm assignment and retained launch logs.",
    )
    parser.add_argument(
        "--endpoint-timeout",
        type=int,
        default=600,
        help="Seconds to wait for endpoint readiness per case.",
    )
    parser.add_argument(
        "--cleanup-timeout",
        type=int,
        default=120,
        help="Seconds to wait after TERM and again after any KILL fallback.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Leave Slurm jobs running for manual inspection.",
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Skip the standard text/vision/full matrix and run only lifecycle checks.",
    )
    parser.add_argument(
        "--lifecycle-case",
        choices=sorted(CASE_MATRIX),
        help="Run a reuse/cleanup/restart lifecycle check for one case after the matrix.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    site_dir = Path(args.site_dir).resolve()
    site_config_path = site_dir / "site.yaml"
    slurm_script_path = site_dir / "serve-model.slurm"

    if not site_config_path.is_file():
        raise FileNotFoundError(f"site.yaml not found: {site_config_path}")
    if not slurm_script_path.is_file():
        raise FileNotFoundError(f"serve-model.slurm not found: {slurm_script_path}")

    os.environ["UMRS_SITE_CONFIG"] = str(site_config_path)
    site = load_site_config(str(site_config_path))
    backend = SlurmPodmanBackend(site=site, slurm_script=str(slurm_script_path))

    selected_cases = [CASE_MATRIX[name] for name in (args.cases or ["text", "vision", "full"])]
    log(f"Using site config {site_config_path}")
    log(f"Using Slurm wrapper {slurm_script_path}")
    log(f"Retained Slurm logs: {backend._output_dir}")

    if not args.skip_matrix:
        for case in selected_cases:
            await run_case(
                backend=backend,
                site=site,
                case=case,
                assignment_timeout_sec=args.assignment_timeout,
                endpoint_timeout_sec=args.endpoint_timeout,
                cleanup_timeout_sec=args.cleanup_timeout,
                skip_cleanup=args.skip_cleanup,
            )

    if args.lifecycle_case:
        await run_lifecycle_case(
            site=site,
            case=CASE_MATRIX[args.lifecycle_case],
            assignment_timeout_sec=args.assignment_timeout,
            endpoint_timeout_sec=args.endpoint_timeout,
            cleanup_timeout_sec=args.cleanup_timeout,
        )

    log("Harness validation completed successfully")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())