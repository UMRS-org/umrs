"""``embedding-job`` sandbox launcher (ADR-061 profile, D-12 move leg).

KNOWLEDGE_EMBED's model compute (tokenize, chunk, embed) runs in a
throwaway rootless-Podman container instead of the UMRS server process:

- ``--network=none``, read-only root filesystem, ``no-new-privileges``, all
  capabilities dropped — the profile's "model-fetch denied, no egress";
- the pre-staged model directory mounted read-only, Hugging Face offline;
- JSON over stdin/stdout (``umrs.knowledge.embed_worker``) — texts in,
  chunks + vectors out. No credentials, database or object-store access
  crosses the boundary: the server keeps every read and write.

Site opt-in: ``UMRS_EMBEDDING_SANDBOX=podman``. Unset = the in-process lane
(unchanged). A configured-but-failing sandbox raises — never a silent fall
back to in-process compute.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SANDBOX_ENV = "UMRS_EMBEDDING_SANDBOX"
IMAGE_ENV = "UMRS_EMBEDDING_SANDBOX_IMAGE"
MODEL_HOST_DIR_ENV = "UMRS_EMBEDDING_SANDBOX_MODEL_HOST_DIR"
TIMEOUT_ENV = "UMRS_EMBEDDING_SANDBOX_TIMEOUT"
MEMORY_ENV = "UMRS_EMBEDDING_SANDBOX_MEMORY"
CPUS_ENV = "UMRS_EMBEDDING_SANDBOX_CPUS"
_CONTAINER_MODEL_DIR = "/model"

Runner = Callable[..., subprocess.CompletedProcess]


class EmbeddingSandboxError(RuntimeError):
    """The configured sandbox could not produce embeddings (named, never masked)."""


def _default_runner(argv: list[str], *, input_text: str, timeout: float) -> Any:
    return subprocess.run(  # argv list, no shell
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@dataclass(frozen=True)
class SandboxConfig:
    image: str
    # Host directory mounted at /model; "" = the model the image bundles.
    model_host_dir: str
    timeout: float = 600.0
    memory: str = "8g"
    cpus: str = "4"
    podman_bin: str = "podman"

    @property
    def container_model_dir(self) -> str:
        """Where the worker loads the model from inside the sandbox."""
        from .text_embedder import BUNDLED_TEXT_EMBEDDER_DIR

        return _CONTAINER_MODEL_DIR if self.model_host_dir else BUNDLED_TEXT_EMBEDDER_DIR


def sandbox_config_from_env(model_dir: str) -> SandboxConfig | None:
    """The sandbox config when the site opted in, else None (in-process lane).

    Opted in without an image is a configuration error, named.
    """
    if os.environ.get(SANDBOX_ENV, "").strip().lower() != "podman":
        return None
    image = os.environ.get(IMAGE_ENV, "").strip() or os.environ.get(
        "UMRS_STEP_IMAGE", ""
    )
    if not image:
        raise EmbeddingSandboxError(
            f"{SANDBOX_ENV}=podman but no image: set {IMAGE_ENV} (or UMRS_STEP_IMAGE)"
        )
    from .text_embedder import BUNDLED_TEXT_EMBEDDER_DIR

    # The podman host may see the pre-staged model under another path than
    # this (containerised) server does. The bundled model is inside the image
    # the sandbox runs, so it needs no mount at all.
    host_dir = os.environ.get(MODEL_HOST_DIR_ENV, "").strip()
    if not host_dir and model_dir != BUNDLED_TEXT_EMBEDDER_DIR:
        host_dir = model_dir
    return SandboxConfig(
        image=image,
        model_host_dir=host_dir,
        timeout=float(os.environ.get(TIMEOUT_ENV, "600")),
        memory=os.environ.get(MEMORY_ENV, "8g"),
        cpus=os.environ.get(CPUS_ENV, "4"),
    )


def build_argv(config: SandboxConfig) -> list[str]:
    """The ``podman run`` argv of one embedding-job sandbox."""
    return [
        config.podman_bin,
        "run",
        "--rm",
        "-i",
        "--network=none",
        "--read-only",
        "--tmpfs=/tmp",
        "--security-opt=no-new-privileges",
        "--cap-drop=all",
        f"--memory={config.memory}",
        f"--cpus={config.cpus}",
        *(["-v", f"{config.model_host_dir}:{_CONTAINER_MODEL_DIR}:ro"] if config.model_host_dir else []),
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "HF_HOME=/tmp/hf",
        config.image,
        "python",
        "-m",
        "umrs.knowledge.embed_worker",
    ]


class SandboxChunkEmbedder:
    """Chunk + embed through one sandbox container per batch of texts."""

    lane = "sandbox"

    def __init__(
        self,
        config: SandboxConfig,
        *,
        prefix_model_name: str,
        max_tokens: int,
        runner: Runner | None = None,
    ) -> None:
        self._config = config
        self._prefix = prefix_model_name
        self._max_tokens = max_tokens
        self._runner = runner or _default_runner

    def chunk_and_embed(self, texts: list[str]) -> list[dict[str, Any]]:
        request = json.dumps(
            {
                "model_dir": self._config.container_model_dir,
                "prefix_model_name": self._prefix,
                "max_tokens": self._max_tokens,
                "texts": texts,
            }
        )
        try:
            done = self._runner(
                build_argv(self._config),
                input_text=request,
                timeout=self._config.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise EmbeddingSandboxError(
                f"embedding sandbox timed out after {self._config.timeout:.0f}s"
            ) from exc
        return _parse_reply(done, expected=len(texts))


def _parse_reply(done: Any, *, expected: int) -> list[dict[str, Any]]:
    try:
        reply = json.loads(done.stdout or "")
    except ValueError as exc:
        raise EmbeddingSandboxError(
            f"embedding sandbox exit={done.returncode}: unreadable reply; "
            f"stderr={str(done.stderr or '')[-300:]}"
        ) from exc
    if done.returncode != 0 or "error" in reply:
        raise EmbeddingSandboxError(
            f"embedding sandbox exit={done.returncode}: {reply.get('error', 'no error text')}"
        )
    results = reply.get("results")
    if not isinstance(results, list) or len(results) != expected:
        raise EmbeddingSandboxError(
            f"embedding sandbox returned {len(results or [])} results for {expected} texts"
        )
    return results


class InProcessChunkEmbedder:
    """The pre-move lane: the server's own embedder (unchanged behaviour)."""

    lane = "in_process"

    def __init__(
        self, embedder: Any, *, prefix_model_name: str, max_tokens: int
    ) -> None:
        self._embedder = embedder
        self._prefix = prefix_model_name
        self._max_tokens = max_tokens

    def chunk_and_embed(self, texts: list[str]) -> list[dict[str, Any]]:
        from .embed_worker import chunk_and_embed

        return chunk_and_embed(
            self._embedder,
            texts,
            prefix_model_name=self._prefix,
            max_tokens=self._max_tokens,
        )


__all__ = [
    "EmbeddingSandboxError",
    "InProcessChunkEmbedder",
    "SandboxChunkEmbedder",
    "SandboxConfig",
    "build_argv",
    "sandbox_config_from_env",
]
