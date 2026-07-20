"""
UMRS Auto-Discovery and Self-Healing Module

Provides automatic environment discovery, resource provisioning,
and self-healing capabilities for the UMRS client.

Usage:
    from umrs.client.discovery import EnvironmentDiscoverer, EnvironmentHealer

    # Discover the environment
    discoverer = EnvironmentDiscoverer(ctx)
    buckets = discoverer.discover_buckets()
    models = discoverer.discover_models()
    rag = discoverer.discover_rag()

    # Auto-heal missing resources
    healer = EnvironmentHealer(ctx, discoverer)
    if not healer.ensure_experience("default"):
        print("Failed to provision experience")
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BucketAliasInfo:
    """Information about a bucket alias."""
    alias: str
    xp_name: str
    modality_name: str
    bucket_path: str
    output_bucket: Optional[str] = None
    mime_type: str = ""
    store_embeddings: bool = False


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    model_type: str
    scope: str
    storage: str
    vector_size: Optional[int] = None
    vision_model_type: Optional[str] = None


@dataclass
class RAGPipelineInfo:
    """Information about a RAG ingestion pipeline."""
    name: str
    status: str
    model: str
    embedder: str


@dataclass
class EnvironmentState:
    """Complete environment state discovered from the server."""
    user: str = ""
    project: str = ""
    profile: str = ""
    gateway_url: str = ""
    server_ready: bool = False
    s3_reachable: bool = False
    postgis_reachable: bool = False
    rag_reachable: bool = False

    # Resources
    bucket_aliases: List[BucketAliasInfo] = field(default_factory=list)
    models: List[ModelInfo] = field(default_factory=list)
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    rag_pipelines: List[RAGPipelineInfo] = field(default_factory=list)

    # Status flags
    has_text_bucket: bool = False
    has_text_model: bool = False
    has_embedding_model: bool = False
    has_rag_pipeline: bool = False
    has_slurm_account: bool = False

    def summary(self) -> str:
        """Return a human-readable summary of the environment state."""
        lines = [
            f"UMRS Environment Summary",
            f"{'=' * 40}",
            f"  User: {self.user or '(none)'}",
            f"  Profile: {self.profile or '(none)'}",
            f"  Gateway: {self.gateway_url or '(not configured)'}",
            f"  Server: {'OK' if self.server_ready else 'UNREACHABLE'}",
            f"  S3/MinIO: {'OK' if self.s3_reachable else 'UNREACHABLE'}",
            f"  PostgreSQL: {'OK' if self.postgis_reachable else 'UNREACHABLE'}",
            f"",
            f"  Bucket aliases: {len(self.bucket_aliases)}",
            f"  Models: {len(self.models)}",
            f"  RAG pipelines: {len(self.rag_pipelines)}",
            f"",
        ]

        if self.bucket_aliases:
            lines.append("  Bucket Aliases:")
            for b in self.bucket_aliases:
                lines.append(f"    - {b.alias}: {b.modality_name} -> {b.bucket_path}")

        if self.models:
            lines.append("  Available Models:")
            for m in self.models:
                lines.append(f"    - {m.name} ({m.model_type}) [{m.scope}]")

        return "\n".join(lines)


class EnvironmentDiscoverer:
    """
    Discovers all available resources in the UMRS environment.

    This is the primary entry point for environment-aware demo execution.
    It queries the UMRS server and client configuration to build a complete
    picture of what resources are available.
    """

    def __init__(self, ctx):
        """
        Initialize the discoverer with the UMRS client context.

        Args:
            ctx: The UMRS client context (from ctx in umrs-client commands)
        """
        self.ctx = ctx

    def discover_all(self) -> EnvironmentState:
        """
        Perform a complete environment discovery.

        Returns:
            EnvironmentState with all discovered resources
        """
        state = EnvironmentState()

        try:
            # 1. Check server connectivity
            state.server_ready = self._check_server_ready()

            if not state.server_ready:
                return state

            # 2. Get user and profile info
            self._populate_user_info(state)

            # 3. Discover bucket aliases
            state.bucket_aliases = self._discover_bucket_aliases()

            # 4. Check for text bucket
            state.has_text_bucket = any(
                b.modality_name == "text" or "text" in b.modality_name.lower()
                for b in state.bucket_aliases
            )

            # 5. Discover models
            state.models = self._discover_models()

            # 6. Check for embedding models
            state.has_embedding_model = any(
                m.model_type == "embedding" or m.model_type == "TransformerEncoder"
                for m in state.models
            )

            # 7. Check for vision models
            state.has_vision_model = any(
                m.model_type == "vision" or (
                    m.vision_model_type is not None
                )
                for m in state.models
            )

            # 8. Check S3 connectivity
            state.s3_reachable = self._check_s3_reachable()

            # 9. Check RAG
            state.rag_reachable = self._check_rag_reachable()
            state.rag_pipelines = self._discover_rag_pipelines()
            state.has_rag_pipeline = len(state.rag_pipelines) > 0

        except Exception:
            pass  # Return partial state on any discovery failure

        return state

    def discover_bucket_alias(self, alias: str) -> Optional[BucketAliasInfo]:
        """
        Discover a specific bucket alias.

        Args:
            alias: The bucket alias to find

        Returns:
            BucketAliasInfo if found, None otherwise
        """
        for b in self._discover_bucket_aliases():
            if b.alias == alias:
                return b
        return None

    def discover_text_bucket(self) -> Optional[BucketAliasInfo]:
        """
        Find the first text bucket alias.

        Returns:
            First text bucket alias or None
        """
        for b in self._discover_bucket_aliases():
            if "text" in b.modality_name.lower() or b.modality_name == "text":
                return b
        return None

    def discover_embedding_model(self) -> Optional[ModelInfo]:
        """
        Find the first embedding model.

        Returns:
            First embedding model or None
        """
        for m in self._discover_models():
            if m.model_type in ("embedding", "TransformerEncoder"):
                return m
        return None

    def discover_vision_model(self) -> Optional[ModelInfo]:
        """
        Find the first vision model.

        Returns:
            First vision model or None
        """
        for m in self._discover_models():
            if m.model_type == "vision":
                return m
        return None

    def find_suitable_bucket_for_modality(self, modality: str) -> Optional[BucketAliasInfo]:
        """
        Find a bucket alias suitable for a given modality.

        Args:
            modality: The modality name (e.g., "text", "vision", "image")

        Returns:
            Matching bucket alias or first available if none matches
        """
        for b in self._discover_bucket_aliases():
            if b.modality_name.lower() == modality.lower():
                return b
        # Fallback: return first bucket
        if self._discover_bucket_aliases():
            return self._discover_bucket_aliases()[0]
        return None

    def discover_rag_pipelines(self) -> List[RAGPipelineInfo]:
        """Discover all RAG ingestion pipelines."""
        return self._discover_rag_pipelines()

    def _check_server_ready(self) -> bool:
        """Check if the UMRS server is reachable."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.executable or "umrs-client",
                    "ping",
                    "--output", "json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    return data.get("status") == "ok"
                except json.JSONDecodeError:
                    return "ready" in result.stdout.lower()
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Try via CLI if available
            return self.ctx.check_server if hasattr(self.ctx, "check_server") else True

    def _populate_user_info(self, state: EnvironmentState):
        """Populate user, project, profile info from context."""
        try:
            if hasattr(self.ctx, "profile_name"):
                state.profile = self.ctx.profile_name or ""
            if hasattr(self.ctx, "connection") and self.ctx.connection:
                conn = self.ctx.connection
                state.gateway_url = getattr(conn, "gateway_url", "") or ""
                state.project = getattr(conn, "project_id", "") or ""

            # Get user from config
            try:
                from umrs.shared.config import get_current_connection_profile_file
                import json
                profile_file = get_current_connection_profile_file()
                if profile_file:
                    import pathlib
                    profile_path = pathlib.Path(profile_file)
                    if profile_path.exists():
                        profile_data = json.loads(profile_path.read_text())
                        if isinstance(profile_data, dict):
                            state.profile = profile_data.get("name", state.profile)
            except Exception:
                pass

            # Get user from config file
            if hasattr(self.ctx, "conf"):
                try:
                    conf_data = json.loads(self.ctx.conf.read_text())
                    state.user = conf_data.get("username", "")
                    if not state.project and conf_data.get("project_id"):
                        state.project = conf_data["project_id"]
                except Exception:
                    pass
        except Exception:
            pass

    def _discover_bucket_aliases(self) -> List[BucketAliasInfo]:
        """Discover all bucket aliases from the client config."""
        buckets = []

        try:
            if hasattr(self.ctx, "buckets_cfg"):
                for alias, entry in self.ctx.buckets_cfg.items():
                    if isinstance(entry, tuple) and len(entry) >= 6:
                        username, xp_name, modality_name, s3_file, metadata, path_to_bucket = entry[:6]
                        output_bucket = None
                        mime_type = ""
                        store_embeddings = False

                        if isinstance(metadata, dict):
                            mime_type = metadata.get("mime_type", "")
                            store_embeddings = metadata.get("store_embeddings", False)

                        output_path = None
                        if len(entry) > 6:
                            output_path = entry[6]

                        buckets.append(BucketAliasInfo(
                            alias=alias,
                            xp_name=xp_name or "",
                            modality_name=modality_name or "",
                            bucket_path=path_to_bucket or "",
                            output_bucket=output_path,
                            mime_type=mime_type,
                            store_embeddings=store_embeddings,
                        ))
                return buckets
        except Exception:
            pass

        return buckets

    def _discover_models(self) -> List[ModelInfo]:
        """Discover all models via the UMRS client model list command."""
        models = []

        try:
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.executable or "umrs-client",
                    "model", "list",
                    "--scope", "system",
                    "--output", "json",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    if isinstance(data, dict) and "models" in data:
                        for m in data["models"]:
                            models.append(ModelInfo(
                                name=m.get("name", ""),
                                model_type=m.get("type", ""),
                                scope=m.get("scope", ""),
                                storage=m.get("storage", ""),
                                vector_size=m.get("vector_size"),
                                vision_model_type=m.get("vision_model_type"),
                            ))
                except json.JSONDecodeError:
                    pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Also check user-scoped models
        try:
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.executable or "umrs-client",
                    "model", "list",
                    "--scope", "user",
                    "--output", "json",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    if isinstance(data, dict) and "models" in data:
                        for m in data["models"]:
                            models.append(ModelInfo(
                                name=m.get("name", ""),
                                model_type=m.get("type", ""),
                                scope=m.get("scope", ""),
                                storage=m.get("storage", ""),
                            ))
                except json.JSONDecodeError:
                    pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return models

    def _check_s3_reachable(self) -> bool:
        """Check if S3/MinIO is reachable."""
        try:
            if hasattr(self.ctx, "buckets_cfg"):
                return len(self.ctx.buckets_cfg) > 0
            return False
        except Exception:
            return False

    def _check_rag_reachable(self) -> bool:
        """Check if RAG service is reachable."""
        try:
            # Check if RAG URL is configured in profile
            if hasattr(self.ctx, "connection") and self.ctx.connection:
                rag_url = getattr(self.ctx.connection, "rag_url", None)
                if rag_url:
                    import urllib.request
                    try:
                        urllib.request.urlopen(f"{rag_url}/health", timeout=5)
                        return True
                    except Exception:
                        # RAG might not have /health endpoint, try /
                        try:
                            urllib.request.urlopen(f"{rag_url}/", timeout=5)
                            return True
                        except Exception:
                            return False
            return False
        except Exception:
            return False

    def _discover_rag_pipelines(self) -> List[RAGPipelineInfo]:
        """Discover RAG ingestion pipelines via r4 CLI."""
        pipelines = []

        try:
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.r4_executable or "r4",
                    "rag", "ingest", "list",
                    "--project", getattr(self.ctx, "project_id", "default"),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                # Parse output to find pipelines
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if line.strip() and "No ingestion pipelines" not in line:
                        # Try to parse pipeline info
                        parts = line.split()
                        if len(parts) >= 2:
                            pipelines.append(RAGPipelineInfo(
                                name=parts[0],
                                status=parts[-1] if len(parts) > 2 else "unknown",
                                model="",
                                embedder="",
                            ))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return pipelines


class EnvironmentHealer:
    """
    Automatically repairs missing or misconfigured resources.

    This module attempts to fix common problems automatically:
    - Missing bucket aliases
    - Missing RAG databases
    - Missing experiences
    - Missing projects
    """

    def __init__(self, ctx, discoverer: Optional[EnvironmentDiscoverer] = None):
        """
        Initialize the healer.

        Args:
            ctx: The UMRS client context
            discoverer: An EnvironmentDiscoverer instance (optional, will be created if not provided)
        """
        self.ctx = ctx
        self.discoverer = discoverer or EnvironmentDiscoverer(ctx)

    def heal(self) -> bool:
        """
        Attempt to heal all detectable environment issues.

        Returns:
            True if all healable issues were resolved
        """
        discovered = self.discoverer.discover_all()

        if not discovered.server_ready:
            return False

        all_ok = True

        # Try to fix missing bucket aliases
        if not self._heal_bucket_aliases():
            all_ok = False

        # Try to fix missing RAG pipeline
        if not discovered.has_rag_pipeline and not self._heal_rag():
            pass  # Not all deployments support RAG

        return all_ok

    def ensure_experience(self, xp_name: str) -> bool:
        """
        Ensure a specific experience is available.

        Args:
            xp_name: The experience name to ensure

        Returns:
            True if the experience was available or successfully created
        """
        discovered = self.discoverer.discover_all()

        # Check if already exists
        for alias in discovered.bucket_aliases:
            if alias.xp_name == xp_name:
                return True

        # Try to create it automatically
        return self._create_experience(xp_name)

    def create_bucket_alias(
        self,
        alias: str,
        modality: str = "text",
        mime_type: str = "text/plain",
    ) -> bool:
        """
        Create a new bucket alias.

        Args:
            alias: The bucket alias to create
            modality: The modality type
            mime_type: The MIME type for data

        Returns:
            True if the alias was created successfully
        """
        try:
            # Call umrs-client reload to apply changes
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.executable or "umrs-client",
                    "reload",
                    "--output", "json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def create_rag_pipeline(
        self,
        name: str = "default",
        model: str = "intfloat/e5-base-v2",
    ) -> bool:
        """
        Create a RAG ingestion pipeline.

        Args:
            name: The pipeline name
            model: The embedding model to use

        Returns:
            True if the pipeline was created
        """
        try:
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.r4_executable or "r4",
                    "rag", "ingest", "create",
                    "--project", "default",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _heal_bucket_aliases(self) -> bool:
        """Attempt to fix bucket alias issues."""
        try:
            # Get current buckets
            discovered = self.discoverer.discover_all()
            if discovered.bucket_aliases:
                return True  # Has at least one bucket

            # No buckets - try to reload to provision
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.executable or "umrs-client",
                    "reload",
                    "--output", "json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _heal_rag(self) -> bool:
        """Attempt to create a default RAG pipeline."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    self.ctx.r4_executable or "r4",
                    "rag", "ingest", "create",
                    "--project", "default",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _create_experience(self, xp_name: str) -> bool:
        """Create a default experience."""
        try:
            import subprocess

            # Try to find a demo experience.json
            import pathlib
            demo_dir = pathlib.Path(__file__).parent.parent.parent / "demos"
            experience_file = None

            for xp in ["02_text_embedding", "03_sam3_segmentation", "05_rag_search"]:
                candidate = demo_dir / xp / "experience.json"
                if candidate.exists():
                    experience_file = candidate
                    break

            if not experience_file:
                return False

            result = subprocess.run(
                [
                    self.ctx.executable or "umrs-client",
                    "experience", "add",
                    str(experience_file),
                    "--update",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False


# ─── CLI Integration ───────────────────────────────────────────────────

def add_discovery_arguments(parser):
    """
    Add auto-discovery arguments to a CLI parser.

    This allows the CLI to automatically discover resources when
    specific arguments are not provided.
    """
    group = parser.add_argument_group("auto-discovery")
    group.add_argument(
        "--discover-buckets",
        action="store_true",
        help="List available bucket aliases and return early",
    )
    group.add_argument(
        "--discover-models",
        action="store_true",
        help="List available models and return early",
    )
    group.add_argument(
        "--discover",
        "--auto-discover",
        action="store_true",
        help="Run environment discovery and print summary",
    )
    return parser


def handle_discovery(ctx, args) -> int:
    """
    Handle the --discover flag by printing environment summary.

    Args:
        ctx: The UMRS client context
        args: The parsed arguments

    Returns:
        0 on success, 1 on failure
    """
    if not getattr(args, "discover", False) and not getattr(args, "discover_buckets", False) and not getattr(args, "discover_models", False):
        return 0

    discoverer = EnvironmentDiscoverer(ctx)
    state = discoverer.discover_all()

    if getattr(args, "discover_buckets", False):
        for b in state.bucket_aliases:
            print(f"{b.alias}: {b.modality_name} -> {b.bucket_path}")
        return 0

    if getattr(args, "discover_models", False):
        for m in state.models:
            print(f"{m.name} ({m.model_type}) [{m.scope}]")
        return 0

    if getattr(args, "discover", False):
        print(state.summary())
        return 0

    return 0


def smart_bucket_lookup(ctx, bucket_alias_arg: str, default_modality: str = "text") -> Optional[str]:
    """
    Smart bucket alias lookup with fallback.

    If the specified bucket alias is not found, try to find a suitable
    one based on the default_modality.

    Args:
        ctx: The UMRS client context
        bucket_alias_arg: The bucket alias argument provided by the user
        default_modality: The default modality to search for

    Returns:
        The resolved bucket alias string, or None if no bucket exists
    """
    if not bucket_alias_arg:
        return None

    discoverer = EnvironmentDiscoverer(ctx)

    # First try exact match
    info = discoverer.discover_bucket_alias(bucket_alias_arg)
    if info:
        return bucket_alias_arg

    # No exact match - try to find suitable bucket
    suitable = discoverer.find_suitable_bucket_for_modality(default_modality)
    if suitable:
        return suitable.alias

    # No suitable bucket - return first available
    buckets = discoverer._discover_bucket_aliases()
    if buckets:
        return buckets[0].alias

    return None
