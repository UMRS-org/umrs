"""Stage the knowledge ingest models (Docling, GLiNER) for offline use (ADR-116 D-9).

UMRS never downloads a model at runtime (offline rule). The server image
build stages both into ``/opt/umrs/models`` (the bundled defaults); a
maintainer may stage other copies once, with network, into the front's
``/models``::

    python -m umrs.knowledge.ingest_models stage docling --dest /models/docling
    python -m umrs.knowledge.ingest_models stage gliner --dest /models/gliner_medium-v2.1

This module holds only what staging needs: the pins, the staged-directory
checks and the ``stage`` / ``verify`` commands. It imports only the standard
library (docling, huggingface_hub and transformers are imported inside the
functions that need them) and no other umrs module, so Dockerfile.server
copies this one file into its model stage and runs it as a script. The
runtime settings live in ``ingest_models``, so editing a default there never
invalidates the image's model layer (about 1.4 GB of downloads).

- Docling: ``PdfPipelineOptions(artifacts_path=<dir>, do_ocr=False)``. Only the
  layout model (heron, and its ONNX export) and TableFormer are staged, each
  at the revision pinned in ``DOCLING_REVISIONS``; OCR, picture
  classification, code and formula models are not.
- GLiNER: ``GLiNER.from_pretrained(<dir>, local_files_only=True)`` under
  ``HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1``. The backbone's tokenizer and
  config are staged inside the directory (``backbone/``), the tokenizer is
  also saved at the top level, and the backbone config is written into
  ``gliner_config.json``, so nothing names a hub repository at load time and
  the directory still loads after it is moved or mounted elsewhere.

The bundled GLiNER and its backbone are pinned (``GLINER_REVISION``,
``GLINER_BACKBONE_REVISION``): ``stage gliner`` uses the pins for the
default repository unless ``--revision`` or ``--backbone-revision`` is
given. An empty pin means unpinned (the hub's current revision). Docling's
repositories are always staged at their pins; a repository the installed
docling reads that has no pin is refused (bump the pins with the lock).

Only weights under a permissive licence are staged (apache-2.0, mit) unless
``--allow-licence`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Baked into the umrs server image at build time (UMRS_BUNDLE_INGEST_MODELS=1),
# like the e5 embedder (text_embedder.BUNDLED_TEXT_EMBEDDER_DIR). A path under
# BUNDLED_MODELS_ROOT is inside the image the ingest sandbox runs, so the
# sandbox mounts nothing for it.
BUNDLED_MODELS_ROOT = "/opt/umrs/models"
BUNDLED_DOCLING_DIR = "/opt/umrs/models/docling"
BUNDLED_GLINER_DIR = "/opt/umrs/models/gliner_medium-v2.1"
GLINER_REPO = "urchade/gliner_medium-v2.1"
# Pinned hub revisions (40-hex commit shas) of GLINER_REPO and of the
# backbone its gliner_config.json names (microsoft/deberta-v3-base); "" would
# mean unpinned. Looked up on 2026-11-13 from
# https://huggingface.co/api/models/<repo> (field "sha"); the same commits the
# 2026-11-12 trial staged. Dockerfile.server stages these and fails the build
# when the staged marker records anything else (``verify``).
GLINER_REVISION = "40ec419335d09393f298636f471328b722c6da9e"
GLINER_BACKBONE_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEFAULT_GLINER_REPO = GLINER_REPO
# The repositories docling 2.150 reads with OCR off, at the commits staged on
# 2026-11-13 (heron and heron-onnx follow upstream "main" in docling's own
# downloader; docling-models is its tag v2.3.0). Bump with the docling lock.
DOCLING_REVISIONS: dict[str, str] = {
    "docling-project/docling-layout-heron": "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
    "docling-project/docling-layout-heron-onnx": "40bde044036bb181c130ddf6c51792187268748f",
    "docling-project/docling-models": "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
}

PERMISSIVE_LICENCES = frozenset({"apache-2.0", "mit"})
MARKER = ".umrs-staged.json"

# Sub-directories docling 2.150 reads for the layout model (heron) and
# TableFormer; used when a directory carries no marker file.
DOCLING_SUBDIRS = (
    "docling-project--docling-layout-heron",
    "docling-project--docling-models",
)
# TableFormer's folder and repository (docling's TableStructureModel).
TABLEFORMER_FOLDER = "docling-project--docling-models"

GLINER_CONFIG = "gliner_config.json"
GLINER_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
BACKBONE_DIR = "backbone"
# Config and tokenizer files of a backbone; never its weights (GLiNER's own
# weights include the encoder).
BACKBONE_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
]


class StagingError(RuntimeError):
    """Staging failed or produced a directory that would not load offline."""


class LicenceError(StagingError):
    """The repository's licence is not one this module stages by default."""


# ── staged-directory checks ─────────────────────────────────────────────────


def _read_marker(path: Path) -> dict[str, Any]:
    try:
        data = json.loads((path / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_marker(path: Path, data: dict[str, Any]) -> None:
    (path / MARKER).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _has_file(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() for p in path.rglob("*"))


def is_docling_staged(model_dir: str | os.PathLike[str]) -> bool:
    """A directory Docling can use as ``artifacts_path`` with OCR off."""
    path = Path(model_dir)
    if not path.is_dir():
        return False
    subdirs = _read_marker(path).get("subdirs") or list(DOCLING_SUBDIRS)
    return all(_has_file(path / str(name)) for name in subdirs)


def is_gliner_staged(model_dir: str | os.PathLike[str]) -> bool:
    """A directory GLiNER loads from with ``local_files_only=True``, offline."""
    path = Path(model_dir)
    if not (path / GLINER_CONFIG).is_file():
        return False
    if not any((path / name).is_file() for name in GLINER_WEIGHTS):
        return False
    return (path / "tokenizer_config.json").is_file() and (
        path / BACKBONE_DIR / "config.json"
    ).is_file()


def gliner_revision(model_dir: str) -> str:
    """The revision a staged directory records, else the pinned one ("" = unpinned)."""
    recorded = _read_marker(Path(model_dir)).get("revision") if model_dir else None
    return str(recorded) if recorded else GLINER_REVISION


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return ""


# ── Docling ─────────────────────────────────────────────────────────────────


def _folder(repo: str) -> str:
    return repo.replace("/", "--")


def docling_repos() -> dict[str, str]:
    """``{folder: repo}`` the installed docling reads with OCR off: every
    variant of its layout model (as its own downloader lists them) and
    TableFormer."""
    try:
        from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
    except ImportError as exc:
        raise StagingError(f"docling is not installed ({exc})") from exc
    spec = LayoutObjectDetectionOptions().model_spec
    repos = {spec.repo_id}
    for override in spec.engine_overrides.values():
        repos.add(str(override.merge_with(spec.repo_id, spec.revision).repo_id))
    repos.add(TABLEFORMER_FOLDER.replace("--", "/"))
    return {_folder(repo): repo for repo in sorted(repos)}


def docling_pins(
    repos: Mapping[str, str], pins: Mapping[str, str] | None = None
) -> dict[str, str]:
    """``{repo: pinned sha}`` for every repository, or StagingError naming the unpinned."""
    table = DOCLING_REVISIONS if pins is None else pins
    missing = sorted(repo for repo in repos.values() if not table.get(repo))
    if missing:
        raise StagingError(
            f"docling reads {missing} with no pinned revision; add it to "
            "DOCLING_REVISIONS (ingest_models_stage.py) with the docling lock bump"
        )
    return {repo: table[repo] for repo in repos.values()}


def _snapshot(repo: str, dest: Path, revision: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=str(dest), revision=revision)


def stage_docling(
    dest: str | os.PathLike[str],
    *,
    download: Callable[[str, Path, str], None] | None = None,
    repos: Mapping[str, str] | None = None,
) -> Path:
    """Download Docling's layout and TableFormer repositories into ``dest``
    at their pinned revisions (needs network)."""
    path = Path(dest)
    path.mkdir(parents=True, exist_ok=True)
    wanted = dict(docling_repos() if repos is None else repos)
    pins = docling_pins(wanted)
    fetch = download or _snapshot
    for folder, repo in wanted.items():
        fetch(repo, path / folder, pins[repo])
    _write_marker(
        path,
        {
            "kind": "docling",
            "docling": _dist_version("docling"),
            "subdirs": sorted(f for f in wanted if _has_file(path / f)),
            "revisions": pins,
        },
    )
    if not is_docling_staged(path) or not all(_has_file(path / f) for f in wanted):
        raise StagingError(f"docling staged into {path} without {sorted(wanted)}")
    return path


# ── GLiNER ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RepoInfo:
    sha: str
    licences: tuple[str, ...]
    files: tuple[str, ...]


class Hub(Protocol):
    def info(self, repo: str, revision: str | None) -> RepoInfo: ...

    def download(
        self, repo: str, dest: Path, revision: str | None, patterns: list[str]
    ) -> None: ...


def _licences_from(card_licence: Any, tags: Iterable[str]) -> tuple[str, ...]:
    if isinstance(card_licence, str):
        values = [card_licence]
    elif isinstance(card_licence, (list, tuple)):
        values = [str(v) for v in card_licence]
    else:
        values = [t.split(":", 1)[1] for t in tags if t.startswith("license:")]
    return tuple(v.strip().lower() for v in values if v and v.strip())


class HuggingFaceHub:
    """The real hub (huggingface_hub, imported on use)."""

    def info(self, repo: str, revision: str | None) -> RepoInfo:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision)
        card = getattr(info, "card_data", None)
        licence = card.get("license") if card is not None else None
        files = tuple(s.rfilename for s in info.siblings or ())
        return RepoInfo(
            sha=info.sha or "",
            licences=_licences_from(licence, info.tags or ()),
            files=files,
        )

    def download(
        self, repo: str, dest: Path, revision: str | None, patterns: list[str]
    ) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo, local_dir=str(dest), revision=revision, allow_patterns=patterns
        )


def _save_tokenizer(source: Path, dest: Path) -> None:
    """Load the backbone tokenizer offline and save it at ``dest``'s top level.

    A DeBERTa-v3 backbone ships only ``spm.model``; converting it needs
    ``sentencepiece`` and ``protobuf``. Any failure is a StagingError, so the
    CLI names it instead of printing a traceback.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        tokenizer.save_pretrained(str(dest))
    except (ImportError, OSError, ValueError) as exc:
        raise StagingError(
            f"the backbone tokenizer in {source} could not be converted "
            f"({type(exc).__name__}: {str(exc)[:300]}); the server image needs "
            "transformers, sentencepiece and protobuf (umrs[kg-ingest])"
        ) from exc


def check_licence(repo: str, info: RepoInfo, *, allow: bool) -> str:
    """The licence string to record; raises LicenceError unless permissive or allowed."""
    licence = ",".join(info.licences) or "unknown"
    permissive = bool(info.licences) and set(info.licences) <= PERMISSIVE_LICENCES
    if not permissive and not allow:
        raise LicenceError(
            f"{repo} licence is {licence!r}, not one of {sorted(PERMISSIVE_LICENCES)}; "
            "pass --allow-licence only after checking the terms"
        )
    return licence


def gliner_weight_pattern(files: Iterable[str]) -> str:
    """``model.safetensors`` when the repo has it, else ``pytorch_model.bin``."""
    names = set(files)
    for name in GLINER_WEIGHTS:
        if name in names:
            return name
    raise StagingError(
        f"no GLiNER weights ({' or '.join(GLINER_WEIGHTS)}) in the repository"
    )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StagingError(f"{path} is not a JSON object")
    return data


def backbone_repo(config: dict[str, Any], marker: dict[str, Any]) -> str:
    """The hub id of the encoder GLiNER was trained on (``model_name``)."""
    name = str(marker.get("backbone") or config.get("model_name") or "").strip()
    if not name or name.startswith(("/", ".")):
        raise StagingError(
            f"{GLINER_CONFIG} names no backbone hub id (model_name={name!r})"
        )
    return name


def offline_gliner_config(
    config: dict[str, Any], backbone_config: dict[str, Any], model_name: str
) -> dict[str, Any]:
    """``config`` with the backbone config inlined and ``model_name`` pointed at the staged copy.

    An ``encoder_config`` the repository already carries is kept as it is.
    """
    out = dict(config)
    out["model_name"] = model_name
    if isinstance(config.get("encoder_config"), dict):
        return out
    encoder = dict(backbone_config)
    encoder.setdefault("model_type", "deberta-v2")
    vocab = config.get("vocab_size", -1)
    if isinstance(vocab, int) and vocab > 0:
        encoder["vocab_size"] = vocab
    out["encoder_config"] = encoder
    return out


def _fetch_gliner(
    hub: Hub, repo: str, revision: str | None, path: Path, allow: bool
) -> dict[str, Any]:
    info = hub.info(repo, revision)
    licence = check_licence(repo, info, allow=allow)
    weights = gliner_weight_pattern(info.files)
    hub.download(
        repo, path, info.sha or revision, [GLINER_CONFIG, weights, "README.md"]
    )
    return {
        "repo": repo,
        "revision": info.sha or revision or "",
        "licence": licence,
        "weights": weights,
    }


def _fetch_backbone(
    hub: Hub, backbone: str, path: Path, allow: bool, revision: str | None = None
) -> dict[str, Any]:
    info = hub.info(backbone, revision)
    licence = check_licence(backbone, info, allow=allow)
    hub.download(
        backbone, path / BACKBONE_DIR, info.sha or revision or None, BACKBONE_FILES
    )
    if not (path / BACKBONE_DIR / "config.json").is_file():
        raise StagingError(f"backbone {backbone} staged without config.json")
    return {
        "backbone": backbone,
        "backbone_revision": info.sha,
        "backbone_licence": licence,
    }


def stage_gliner(
    dest: str | os.PathLike[str],
    *,
    repo: str = DEFAULT_GLINER_REPO,
    revision: str | None = None,
    backbone_revision: str | None = None,
    allow_licence: bool = False,
    mount_path: str | None = None,
    hub: Hub | None = None,
    save_tokenizer: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Download a GLiNER model and its backbone tokenizer/config into ``dest`` (needs network).

    ``revision`` / ``backbone_revision`` (None = the hub's current one) pin
    the two repositories. ``mount_path`` is where the directory will be seen
    at load time (for example ``/models/gliner_medium-v2.1`` inside the
    server container); it defaults to ``dest``. The loader does not need it,
    because the backbone config is inlined and the tokenizer sits at the
    top level.
    """
    path = Path(dest).resolve()
    path.mkdir(parents=True, exist_ok=True)
    client = hub or HuggingFaceHub()
    marker = _fetch_gliner(client, repo, revision, path, allow_licence)
    config = _read_json(path / GLINER_CONFIG)
    backbone = backbone_repo(config, _read_marker(path))
    marker.update(
        _fetch_backbone(client, backbone, path, allow_licence, backbone_revision)
    )
    (save_tokenizer or _save_tokenizer)(path / BACKBONE_DIR, path)
    model_name = str(Path(mount_path or path) / BACKBONE_DIR)
    backbone_config = _read_json(path / BACKBONE_DIR / "config.json")
    rewritten = offline_gliner_config(config, backbone_config, model_name)
    (path / GLINER_CONFIG).write_text(
        json.dumps(rewritten, indent=2) + "\n", encoding="utf-8"
    )
    _write_marker(path, {"kind": "gliner", **marker})
    if not is_gliner_staged(path):
        raise StagingError(f"{repo} staged into {path} but the directory is incomplete")
    return path


# ── pin verification (the image build) ──────────────────────────────────────


def pin_mismatches(docling_dir: str, gliner_dir: str) -> list[str]:
    """What a staged pair records against the pins; [] when every pin matches."""
    bad: list[str] = []
    recorded = _read_marker(Path(docling_dir)).get("revisions") or {}
    for repo, sha in DOCLING_REVISIONS.items():
        if recorded.get(repo) != sha:
            bad.append(f"{repo}: {recorded.get(repo)!r} != {sha}")
    marker = _read_marker(Path(gliner_dir))
    for key, sha in (
        ("revision", GLINER_REVISION),
        ("backbone_revision", GLINER_BACKBONE_REVISION),
    ):
        if sha and marker.get(key) != sha:
            bad.append(f"gliner {key}: {marker.get(key)!r} != {sha}")
    return bad


# ── CLI ─────────────────────────────────────────────────────────────────────


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def add_stage_commands(sub: Any) -> None:
    """The ``stage`` and ``verify`` sub-commands (also used by ``ingest_models``)."""
    stage = sub.add_parser(
        "stage", help="download a model for offline use (needs network)"
    )
    which = stage.add_subparsers(dest="model", required=True)
    p_doc = which.add_parser("docling", help="Docling layout + TableFormer weights")
    p_doc.add_argument("--dest", required=True)
    p_gl = which.add_parser(
        "gliner", help="a GLiNER model with its backbone tokenizer/config"
    )
    p_gl.add_argument("--dest", required=True)
    p_gl.add_argument("--repo", default=GLINER_REPO)
    p_gl.add_argument(
        "--revision", default=None, help=f"default: the pin for {GLINER_REPO}"
    )
    p_gl.add_argument(
        "--backbone-revision", default=None, help="default: the backbone's pin"
    )
    p_gl.add_argument(
        "--mount-path", default=None, help="the directory's path at load time"
    )
    p_gl.add_argument(
        "--allow-licence",
        action="store_true",
        help="stage a non apache-2.0/mit repository",
    )
    verify = sub.add_parser("verify", help="check staged directories against the pins")
    verify.add_argument("--docling", required=True)
    verify.add_argument("--gliner", required=True)


def pinned_revisions(
    repo: str, revision: str | None, backbone_revision: str | None
) -> tuple[str | None, str | None]:
    """The revisions to stage: an explicit one wins; for ``GLINER_REPO`` an
    unset one is its pin (an empty pin = None, the hub's current revision)."""
    if repo != GLINER_REPO:
        return revision, backbone_revision
    return (
        revision or GLINER_REVISION or None,
        backbone_revision or GLINER_BACKBONE_REVISION or None,
    )


def _stage(args: argparse.Namespace) -> Path:
    if args.model == "docling":
        return stage_docling(args.dest)
    revision, backbone_revision = pinned_revisions(
        args.repo, args.revision, args.backbone_revision
    )
    return stage_gliner(
        args.dest,
        repo=args.repo,
        revision=revision,
        backbone_revision=backbone_revision,
        allow_licence=args.allow_licence,
        mount_path=args.mount_path,
    )


def _verify(args: argparse.Namespace) -> int:
    bad = pin_mismatches(args.docling, args.gliner)
    for line in bad:
        print(f"error: staged off its pin: {line}", file=sys.stderr)
    return 3 if bad else 0


def run_command(args: argparse.Namespace) -> int:
    """Run a parsed ``stage`` or ``verify`` command."""
    if args.command == "verify":
        return _verify(args)
    try:
        path = _stage(args)
    except StagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"staged {args.model} -> {path} ({_dir_size(path) / 1e6:.0f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest_models_stage.py")
    add_stage_commands(parser.add_subparsers(dest="command", required=True))
    return run_command(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
