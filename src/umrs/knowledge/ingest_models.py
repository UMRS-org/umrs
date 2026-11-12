"""Stage the knowledge ingest models (Docling, GLiNER) for offline use (ADR-116 D-9).

UMRS never downloads a model at runtime (offline rule). A maintainer stages
the weights once, with network, into the front's ``/models``::

    python -m umrs.knowledge.ingest_models stage docling --dest /models/docling
    python -m umrs.knowledge.ingest_models stage gliner --dest /models/gliner_medium-v2.1

The understand stage then loads them offline:

- Docling: ``PdfPipelineOptions(artifacts_path=<dir>, do_ocr=False)``. Only the
  layout model and TableFormer are staged; OCR, picture classification, code
  and formula models are not.
- GLiNER: ``GLiNER.from_pretrained(<dir>, local_files_only=True)`` under
  ``HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1``. The backbone's tokenizer and
  config are staged inside the directory (``backbone/``), the tokenizer is
  also saved at the top level, and the backbone config is written into
  ``gliner_config.json``, so nothing names a hub repository at load time and
  the directory still loads after it is moved or mounted elsewhere.

Settings (read by the understand stage):

- ``UMRS_KNOWLEDGE_DOC_PARSER``: ``pypdf`` (default) or ``docling``.
- ``UMRS_KNOWLEDGE_DOCLING_ARTIFACTS``: Docling's staged directory, default
  ``/models/docling``.
- ``UMRS_KNOWLEDGE_GLINER_MODEL``: GLiNER's staged directory; unset or empty
  means no pre-tagging.

Only weights under a permissive licence are staged (apache-2.0, mit) unless
``--allow-licence`` is passed. This module imports only the standard library
at load time; docling, huggingface_hub and transformers are imported inside
the functions that need them.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DOC_PARSER_ENV = "UMRS_KNOWLEDGE_DOC_PARSER"
DOCLING_ARTIFACTS_ENV = "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS"
GLINER_MODEL_ENV = "UMRS_KNOWLEDGE_GLINER_MODEL"
DOC_PARSERS = ("pypdf", "docling")
DEFAULT_DOC_PARSER = "pypdf"
DEFAULT_DOCLING_ARTIFACTS = "/models/docling"
DEFAULT_GLINER_REPO = "urchade/gliner_medium-v2.1"

PERMISSIVE_LICENCES = frozenset({"apache-2.0", "mit"})
MARKER = ".umrs-staged.json"

# Docling download flags turned on; every other ``with_*`` flag of the
# installed docling's download_models() is passed as False.
DOCLING_WANTED = frozenset({"with_layout", "with_tableformer"})
# Sub-directories docling 2.150 writes for the layout model (heron) and
# TableFormer; used when a directory carries no marker file.
DOCLING_SUBDIRS = (
    "docling-project--docling-layout-heron",
    "docling-project--docling-models",
)

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


# ── settings ────────────────────────────────────────────────────────────────


def configured_doc_parser() -> str:
    """``docling`` or ``pypdf``; an unknown value falls back to ``pypdf``."""
    value = os.environ.get(DOC_PARSER_ENV, "").strip().lower()
    return value if value in DOC_PARSERS else DEFAULT_DOC_PARSER


def configured_docling_artifacts() -> str:
    return (
        os.environ.get(DOCLING_ARTIFACTS_ENV, "").strip() or DEFAULT_DOCLING_ARTIFACTS
    )


def configured_gliner_model() -> str:
    """The staged GLiNER directory, or ``""`` (pre-tagging off)."""
    return os.environ.get(GLINER_MODEL_ENV, "").strip()


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


# ── Docling ─────────────────────────────────────────────────────────────────


def _docling_downloader() -> Callable[..., Any]:
    from docling.utils.model_downloader import download_models

    return download_models


def docling_download_flags(download: Callable[..., Any]) -> dict[str, bool]:
    """Layout and TableFormer on, every other ``with_*`` flag off."""
    params = inspect.signature(download).parameters
    missing = DOCLING_WANTED - set(params)
    if missing:
        raise StagingError(f"docling download_models() lacks {sorted(missing)}")
    return {name: name in DOCLING_WANTED for name in params if name.startswith("with_")}


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def stage_docling(
    dest: str | os.PathLike[str], *, download: Callable[..., Any] | None = None
) -> Path:
    """Download Docling's layout and TableFormer weights into ``dest`` (needs network)."""
    path = Path(dest)
    path.mkdir(parents=True, exist_ok=True)
    fetch = download or _docling_downloader()
    fetch(output_dir=path, progress=False, **docling_download_flags(fetch))
    subdirs = sorted(p.name for p in path.iterdir() if _has_file(p))
    _write_marker(
        path,
        {"kind": "docling", "docling": _dist_version("docling"), "subdirs": subdirs},
    )
    if not is_docling_staged(path):
        raise StagingError(
            f"docling staged into {path} without {list(DOCLING_SUBDIRS)}"
        )
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


def _fetch_backbone(hub: Hub, backbone: str, path: Path, allow: bool) -> dict[str, Any]:
    info = hub.info(backbone, None)
    licence = check_licence(backbone, info, allow=allow)
    hub.download(backbone, path / BACKBONE_DIR, info.sha or None, BACKBONE_FILES)
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
    allow_licence: bool = False,
    mount_path: str | None = None,
    hub: Hub | None = None,
    save_tokenizer: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Download a GLiNER model and its backbone tokenizer/config into ``dest`` (needs network).

    ``mount_path`` is where the directory will be seen at load time (for
    example ``/models/gliner_medium-v2.1`` inside the server container); it
    defaults to ``dest``. The loader does not need it, because the backbone
    config is inlined and the tokenizer sits at the top level.
    """
    path = Path(dest).resolve()
    path.mkdir(parents=True, exist_ok=True)
    client = hub or HuggingFaceHub()
    marker = _fetch_gliner(client, repo, revision, path, allow_licence)
    config = _read_json(path / GLINER_CONFIG)
    marker.update(
        _fetch_backbone(
            client, backbone_repo(config, _read_marker(path)), path, allow_licence
        )
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


# ── CLI ─────────────────────────────────────────────────────────────────────


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m umrs.knowledge.ingest_models")
    sub = parser.add_subparsers(dest="command", required=True)
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
    p_gl.add_argument("--repo", default=DEFAULT_GLINER_REPO)
    p_gl.add_argument("--revision", default=None)
    p_gl.add_argument(
        "--mount-path", default=None, help="the directory's path at load time"
    )
    p_gl.add_argument(
        "--allow-licence",
        action="store_true",
        help="stage a non apache-2.0/mit repository",
    )
    sub.add_parser(
        "show",
        help="print the configured parser and models and whether they are staged",
    )
    return parser


def _stage(args: argparse.Namespace) -> Path:
    if args.model == "docling":
        return stage_docling(args.dest)
    return stage_gliner(
        args.dest,
        repo=args.repo,
        revision=args.revision,
        allow_licence=args.allow_licence,
        mount_path=args.mount_path,
    )


def _show() -> int:
    docling_dir = configured_docling_artifacts()
    gliner_dir = configured_gliner_model()
    parser = configured_doc_parser()
    docling_ok = is_docling_staged(docling_dir)
    gliner_ok = not gliner_dir or is_gliner_staged(gliner_dir)
    print(f"{DOC_PARSER_ENV}={parser}")
    print(
        f"{DOCLING_ARTIFACTS_ENV}={docling_dir} ({'staged' if docling_ok else 'NOT staged'})"
    )
    gliner_state = "staged" if gliner_ok else "NOT staged"
    print(
        f"{GLINER_MODEL_ENV}={gliner_dir} ({gliner_state})"
        if gliner_dir
        else f"{GLINER_MODEL_ENV} unset (off)"
    )
    return 0 if gliner_ok and (docling_ok or parser == "pypdf") else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "show":
        return _show()
    try:
        path = _stage(args)
    except StagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"staged {args.model} -> {path} ({_dir_size(path) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
