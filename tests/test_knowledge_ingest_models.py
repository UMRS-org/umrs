"""Staging of the knowledge ingest models (ADR-116 D-9, umrs.knowledge.ingest_models).

Hermetic: the hub, Docling's downloader and the tokenizer save are fakes; no
network, no docling/gliner/transformers import.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from umrs.knowledge import ingest_models as im
from umrs.knowledge import ingest_models_stage as ims

GLINER_CONFIG = {"model_name": "microsoft/deberta-v3-base", "vocab_size": 128004, "max_len": 384}
BACKBONE_CONFIG = {"model_type": "deberta-v2", "hidden_size": 768, "vocab_size": 128100}


class FakeHub:
    """Serves two repos: a GLiNER model and its backbone."""

    def __init__(self, licences: dict[str, tuple[str, ...]] | None = None, gliner_files=None):
        self.licences = licences or {}
        self.gliner_files = gliner_files or (
            "gliner_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        )
        self.downloads: list[tuple[str, str | None, list[str]]] = []
        self.revisions: dict[str, str | None] = {}

    def info(self, repo, revision):
        self.revisions[repo] = revision
        files = self.gliner_files if repo.startswith("urchade/") else ("config.json", "spm.model")
        licence = self.licences.get(
            repo, ("apache-2.0",) if repo.startswith("urchade/") else ("mit",)
        )
        return ims.RepoInfo(sha=f"sha-{repo}", licences=licence, files=files)

    def download(self, repo, dest, revision, patterns):
        self.downloads.append((repo, revision, list(patterns)))
        dest.mkdir(parents=True, exist_ok=True)
        if repo.startswith("urchade/"):
            (dest / "gliner_config.json").write_text(json.dumps(GLINER_CONFIG))
            for name in im.GLINER_WEIGHTS:
                if name in patterns:
                    (dest / name).write_bytes(b"w")
            return
        (dest / "config.json").write_text(json.dumps(BACKBONE_CONFIG))
        (dest / "spm.model").write_bytes(b"s")


def fake_save_tokenizer(source: Path, dest: Path) -> None:
    assert (source / "spm.model").is_file()
    (dest / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "DebertaV2Tokenizer"})
    )


def _stage(tmp_path, hub=None, **kwargs):
    return im.stage_gliner(
        tmp_path / "gliner", hub=hub or FakeHub(), save_tokenizer=fake_save_tokenizer, **kwargs
    )


# ── settings ────────────────────────────────────────────────────────────────


_SETTINGS = (
    im.DOC_PARSER_ENV,
    im.DOCLING_ARTIFACTS_ENV,
    im.GLINER_MODEL_ENV,
    im.DOCMODELS_URL_ENV,
    im.DOCMODELS_TIMEOUT_ENV,
    im.DOCLING_MAX_PAGES_ENV,
    im.DOCLING_MAX_BYTES_ENV,
    im.DOCMODELS_LOCAL_FALLBACK_ENV,
    im.DOCMODELS_TOKEN_ENV,
)


def test_settings_defaults_are_on_and_bundled(monkeypatch):
    """2026-11-13: Docling and GLiNER are on by default, read from the image."""
    for name in _SETTINGS:
        monkeypatch.delenv(name, raising=False)
    assert im.configured_doc_parser() == "docling"
    assert im.configured_docling_artifacts() == "/opt/umrs/models/docling"
    assert im.configured_gliner_dir() == "/opt/umrs/models/gliner_medium-v2.1"
    assert im.configured_docmodels_url() == ""
    assert im.configured_docmodels_timeout() == 120.0
    assert im.configured_docling_max_pages() == 80


def test_review_settings_defaults(monkeypatch):
    """The byte cap both sides read, the sandbox-only fallback, no token."""
    for name in _SETTINGS:
        monkeypatch.delenv(name, raising=False)
    assert im.configured_docling_max_bytes() == 64 * 1024 * 1024
    assert im.configured_local_fallback() == "sandbox"
    assert im.configured_docmodels_token() == ""


def test_settings_values(monkeypatch):
    monkeypatch.setenv(im.DOC_PARSER_ENV, " Docling ")
    monkeypatch.setenv(im.GLINER_MODEL_ENV, "  ")
    assert im.configured_doc_parser() == "docling"
    assert im.configured_gliner_dir() == im.BUNDLED_GLINER_DIR
    monkeypatch.setenv(im.DOC_PARSER_ENV, "marker")
    assert im.configured_doc_parser() == "docling"  # unknown reads as the default
    monkeypatch.setenv(im.GLINER_MODEL_ENV, "/models/g")
    monkeypatch.setenv(im.DOCMODELS_URL_ENV, "http://umrs-docmodels:8091/")
    monkeypatch.setenv(im.DOCMODELS_TIMEOUT_ENV, "bogus")
    assert im.configured_gliner_dir() == "/models/g"
    assert im.configured_docmodels_url() == "http://umrs-docmodels:8091"
    assert im.configured_docmodels_timeout() == 120.0


@pytest.mark.parametrize("value", ["pypdf", "off", "OFF", "none", "false", "0"])
def test_the_parser_off_values(monkeypatch, value):
    monkeypatch.setenv(im.DOC_PARSER_ENV, value)
    assert im.configured_doc_parser() == "pypdf"


@pytest.mark.parametrize("value", ["off", "None", "false", "0"])
def test_the_gliner_and_service_off_values(monkeypatch, value):
    monkeypatch.setenv(im.GLINER_MODEL_ENV, value)
    monkeypatch.setenv(im.DOCMODELS_URL_ENV, value)
    assert im.configured_gliner_dir() == ""
    assert im.configured_docmodels_url() == ""


@pytest.mark.parametrize(
    ("value", "mode"),
    [
        ("inprocess", "inprocess"),
        ("SANDBOX", "sandbox"),
        ("off", "off"),
        ("0", "off"),
        ("bogus", "sandbox"),
    ],
)
def test_the_local_fallback_setting(monkeypatch, value, mode):
    monkeypatch.setenv(im.DOCMODELS_LOCAL_FALLBACK_ENV, value)
    assert im.configured_local_fallback() == mode


def test_document_refusals_are_final():
    """A document the service refused or died on is never parsed again locally."""
    for reason in ("too_large", "too_many_pages", "parse_failed", "tag_failed", "transport_error"):
        assert reason in im.SERVICE_FINAL_REASONS
    for reason in ("busy", "loading", "connect_failed", "untrusted", "model_mismatch"):
        assert reason not in im.SERVICE_FINAL_REASONS


def test_the_staging_module_is_standalone():
    """Dockerfile.server copies only ingest_models_stage.py into its model
    stage: it must import nothing but the standard library and hold no
    runtime setting, so a settings edit never re-downloads the weights."""
    source = Path(ims.__file__).read_text(encoding="utf-8")
    assert _top_level_imports(source) <= set(sys.stdlib_module_names) | {"__future__"}
    assert "configured_" not in source and "UMRS_KNOWLEDGE_" not in source
    out = subprocess.run(
        [sys.executable, ims.__file__, "verify", "--help"], capture_output=True, text=True
    )
    assert out.returncode == 0 and "--docling" in out.stdout


def _top_level_imports(source: str) -> set[str]:
    """Top-level package names a module imports at load ("." for a relative one)."""
    import ast

    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom):
            names.add("." if node.level else (node.module or "").split(".")[0])
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
    return names


def test_the_docmodels_budget_rule(monkeypatch):
    """min(T, deadline + T - now): capped at T, never past deadline + T."""
    monkeypatch.setenv(im.DOCMODELS_TIMEOUT_ENV, "120")
    assert im.docmodels_budget(None) == 120.0
    assert im.docmodels_budget(1000.0, now=500.0) == 120.0  # far from the deadline
    assert im.docmodels_budget(1000.0, now=990.0) == 120.0  # T is outside the LLM budget
    assert im.docmodels_budget(1000.0, now=1050.0) == 70.0  # past it: until deadline + T
    assert im.docmodels_budget(1000.0, now=1120.0) == 0.0


def test_the_pins_are_empty_or_40_hex():
    for pin in (im.GLINER_REVISION, im.GLINER_BACKBONE_REVISION):
        assert pin == "" or (len(pin) == 40 and int(pin, 16) >= 0)
    assert im.GLINER_REPO == "urchade/gliner_medium-v2.1"
    assert im.BUNDLED_DOCLING_DIR.startswith(im.BUNDLED_MODELS_ROOT + "/")
    assert im.BUNDLED_GLINER_DIR.startswith(im.BUNDLED_MODELS_ROOT + "/")


def test_cli_stage_gliner_uses_the_pins_by_default(monkeypatch, tmp_path):
    seen = {}

    def fake_stage(dest, **kwargs):
        seen.update(kwargs)
        Path(dest).mkdir(parents=True)
        return Path(dest)

    monkeypatch.setattr(ims, "stage_gliner", fake_stage)
    monkeypatch.setattr(ims, "GLINER_REVISION", "a" * 40)
    monkeypatch.setattr(ims, "GLINER_BACKBONE_REVISION", "b" * 40)
    assert im.main(["stage", "gliner", "--dest", str(tmp_path / "g")]) == 0
    assert (seen["revision"], seen["backbone_revision"]) == ("a" * 40, "b" * 40)
    argv = ["stage", "gliner", "--dest", str(tmp_path / "h"), "--revision", "c" * 40]
    assert im.main(argv) == 0
    assert (seen["revision"], seen["backbone_revision"]) == ("c" * 40, "b" * 40)


def test_empty_pins_mean_the_hub_revision():
    assert ims.pinned_revisions(im.GLINER_REPO, None, None) == (
        im.GLINER_REVISION or None,
        im.GLINER_BACKBONE_REVISION or None,
    )
    assert ims.pinned_revisions("other/repo", None, None) == (None, None)


def test_stage_gliner_pins_the_backbone_revision(tmp_path):
    hub = FakeHub()
    _stage(tmp_path, hub=hub, backbone_revision="b" * 40)
    backbone = [d for d in hub.downloads if d[0] == "microsoft/deberta-v3-base"]
    assert backbone[0][1] == "sha-microsoft/deberta-v3-base"  # the hub's sha for it
    assert hub.revisions["microsoft/deberta-v3-base"] == "b" * 40


def test_gliner_revision_reads_the_marker(tmp_path):
    path = _stage(tmp_path)
    assert im.gliner_revision(str(path)) == "sha-urchade/gliner_medium-v2.1"
    assert im.gliner_revision("") == im.GLINER_REVISION


def test_module_imports_no_heavy_library():
    code = (
        "import sys, umrs.knowledge.ingest_models; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'docling', 'gliner', 'huggingface_hub', 'transformers', 'torch'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ── Docling ─────────────────────────────────────────────────────────────────


_REPOS = {
    "docling-project--docling-layout-heron": "docling-project/docling-layout-heron",
    "docling-project--docling-models": "docling-project/docling-models",
}


def _fake_snapshot(calls: list):
    def download(repo: str, dest: Path, revision: str) -> None:
        calls.append((repo, dest.name, revision))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.safetensors").write_bytes(b"w")

    return download


def test_stage_docling_downloads_each_repository_at_its_pin(tmp_path):
    """docling's own downloader fetched heron at upstream "main": the public
    image carried whatever was pushed at build time."""
    calls: list = []
    dest = tmp_path / "docling"
    assert not im.is_docling_staged(dest)
    assert im.stage_docling(dest, download=_fake_snapshot(calls), repos=_REPOS) == dest
    assert sorted(calls) == sorted(
        (repo, folder, im.DOCLING_REVISIONS[repo]) for folder, repo in _REPOS.items()
    )
    marker = json.loads((dest / im.MARKER).read_text())
    assert (marker["kind"], marker["subdirs"]) == ("docling", sorted(im.DOCLING_SUBDIRS))
    assert marker["revisions"] == {repo: im.DOCLING_REVISIONS[repo] for repo in _REPOS.values()}
    assert im.is_docling_staged(dest)


def test_an_unpinned_docling_repository_is_refused(tmp_path):
    calls: list = []
    repos = {**_REPOS, "x--new-layout": "x/new-layout"}
    with pytest.raises(im.StagingError, match="x/new-layout"):
        im.stage_docling(tmp_path / "d", download=_fake_snapshot(calls), repos=repos)
    assert calls == []


def test_the_docling_pins_are_40_hex_and_cover_the_subdirs():
    for repo, sha in im.DOCLING_REVISIONS.items():
        assert len(sha) == 40 and int(sha, 16) >= 0, repo
    folders = {repo.replace("/", "--") for repo in im.DOCLING_REVISIONS}
    assert set(im.DOCLING_SUBDIRS) <= folders


def test_stage_docling_fails_when_download_leaves_nothing(tmp_path):
    with pytest.raises(im.StagingError):
        im.stage_docling(tmp_path / "d", download=lambda *_a: None, repos=_REPOS)


def test_docling_repos_reads_the_installed_docling():
    pytest.importorskip("docling")
    repos = ims.docling_repos()
    assert repos["docling-project--docling-models"] == "docling-project/docling-models"
    assert "docling-project--docling-layout-heron" in repos
    assert ims.docling_pins(repos) == {repo: im.DOCLING_REVISIONS[repo] for repo in repos.values()}


def test_verify_names_a_directory_staged_off_its_pins(tmp_path, capsys):
    docling = im.stage_docling(tmp_path / "d", download=_fake_snapshot([]), repos=_REPOS)
    gliner = _stage(
        tmp_path,
        revision=im.GLINER_REVISION,
        backbone_revision=im.GLINER_BACKBONE_REVISION,
    )
    bad = ims.pin_mismatches(str(docling), str(gliner))
    # FakeHub records "sha-<repo>", not the pins; heron-onnx was not staged.
    assert any("heron-onnx" in line for line in bad)
    assert any(line.startswith("gliner revision") for line in bad)
    assert im.main(["verify", "--docling", str(docling), "--gliner", str(gliner)]) == 3
    assert "staged off its pin" in capsys.readouterr().err


def test_verify_passes_a_directory_staged_at_its_pins(tmp_path):
    docling = tmp_path / "d"
    docling.mkdir()
    ims._write_marker(docling, {"revisions": dict(im.DOCLING_REVISIONS)})
    gliner = tmp_path / "g"
    gliner.mkdir()
    ims._write_marker(
        gliner,
        {"revision": im.GLINER_REVISION, "backbone_revision": im.GLINER_BACKBONE_REVISION},
    )
    assert ims.pin_mismatches(str(docling), str(gliner)) == []


def test_is_docling_staged_without_marker(tmp_path):
    for name in im.DOCLING_SUBDIRS:
        (tmp_path / name).mkdir()
    assert not im.is_docling_staged(tmp_path)  # empty sub-directories
    for name in im.DOCLING_SUBDIRS:
        (tmp_path / name / "f").write_bytes(b"x")
    assert im.is_docling_staged(tmp_path)
    assert not im.is_docling_staged(tmp_path / "missing")


# ── GLiNER ──────────────────────────────────────────────────────────────────


def test_stage_gliner_makes_an_offline_directory(tmp_path):
    hub = FakeHub()
    path = _stage(tmp_path, hub=hub, mount_path="/models/gliner")
    assert im.is_gliner_staged(path)
    config = json.loads((path / "gliner_config.json").read_text())
    assert config["model_name"] == "/models/gliner/backbone"
    assert config["encoder_config"]["model_type"] == "deberta-v2"
    assert config["encoder_config"]["vocab_size"] == 128004  # GLiNER's resized vocabulary
    assert config["max_len"] == 384
    marker = json.loads((path / im.MARKER).read_text())
    assert marker["backbone"] == "microsoft/deberta-v3-base" and marker["licence"] == "apache-2.0"
    assert marker["weights"] == "model.safetensors"


def test_stage_gliner_fetches_safetensors_only_and_no_backbone_weights(tmp_path):
    hub = FakeHub()
    _stage(tmp_path, hub=hub)
    (gl_repo, gl_rev, gl_patterns), (bb_repo, _, bb_patterns) = hub.downloads
    assert gl_repo == im.DEFAULT_GLINER_REPO and gl_rev == f"sha-{gl_repo}"
    assert "model.safetensors" in gl_patterns and "pytorch_model.bin" not in gl_patterns
    assert bb_repo == "microsoft/deberta-v3-base"
    assert not any(p.endswith((".bin", ".safetensors", ".h5", ".ot")) for p in bb_patterns)


def test_stage_gliner_falls_back_to_pytorch_bin(tmp_path):
    hub = FakeHub(gliner_files=("gliner_config.json", "pytorch_model.bin"))
    path = _stage(tmp_path, hub=hub)
    assert (path / "pytorch_model.bin").is_file() and im.is_gliner_staged(path)


def test_stage_gliner_refuses_a_non_permissive_licence(tmp_path):
    hub = FakeHub(licences={im.DEFAULT_GLINER_REPO: ("cc-by-nc-4.0",)})
    with pytest.raises(im.LicenceError, match="cc-by-nc-4.0"):
        _stage(tmp_path, hub=hub)
    assert hub.downloads == []


def test_stage_gliner_refuses_an_unknown_licence_unless_allowed(tmp_path):
    hub = FakeHub(licences={im.DEFAULT_GLINER_REPO: ()})
    with pytest.raises(im.LicenceError, match="unknown"):
        _stage(tmp_path, hub=hub)
    path = _stage(tmp_path, hub=hub, allow_licence=True)
    assert json.loads((path / im.MARKER).read_text())["licence"] == "unknown"


def test_stage_gliner_checks_the_backbone_licence(tmp_path):
    hub = FakeHub(licences={"microsoft/deberta-v3-base": ("other",)})
    with pytest.raises(im.LicenceError, match="deberta"):
        _stage(tmp_path, hub=hub)


def test_restage_reads_the_backbone_from_the_marker(tmp_path):
    path = _stage(tmp_path)
    # A re-run whose download leaves the rewritten config in place.
    config = json.loads((path / "gliner_config.json").read_text())
    assert (
        ims.backbone_repo(config, json.loads((path / im.MARKER).read_text()))
        == "microsoft/deberta-v3-base"
    )
    with pytest.raises(im.StagingError):
        ims.backbone_repo(config, {})


def test_offline_config_keeps_a_shipped_encoder_config():
    config = {"model_name": "x/y", "encoder_config": {"model_type": "modernbert"}}
    out = ims.offline_gliner_config(config, BACKBONE_CONFIG, "/m/backbone")
    assert (
        out["encoder_config"] == {"model_type": "modernbert"} and out["model_name"] == "/m/backbone"
    )


def test_is_gliner_staged_needs_every_part(tmp_path):
    assert not im.is_gliner_staged(tmp_path)
    (tmp_path / "gliner_config.json").write_text("{}")
    assert not im.is_gliner_staged(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"w")
    assert not im.is_gliner_staged(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    assert not im.is_gliner_staged(tmp_path)
    (tmp_path / "backbone").mkdir()
    (tmp_path / "backbone" / "config.json").write_text("{}")
    assert im.is_gliner_staged(tmp_path)


def test_licences_from_card_or_tags():
    assert ims._licences_from("Apache-2.0", ()) == ("apache-2.0",)
    assert ims._licences_from(["mit", "apache-2.0"], ()) == ("mit", "apache-2.0")
    assert ims._licences_from(None, ["license:mit", "en"]) == ("mit",)
    assert ims._licences_from(None, []) == ()


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_stage_gliner_passes_options(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_stage(dest, **kwargs):
        seen.update(kwargs, dest=dest)
        Path(dest).mkdir(parents=True)
        return Path(dest)

    monkeypatch.setattr(ims, "stage_gliner", fake_stage)
    dest = str(tmp_path / "g")
    argv = [
        "stage",
        "gliner",
        "--dest",
        dest,
        "--repo",
        "a/b",
        "--allow-licence",
        "--mount-path",
        "/models/g",
    ]
    assert im.main(argv) == 0
    assert seen == {
        "dest": dest,
        "repo": "a/b",
        "revision": None,  # another repo than the pinned one: no pin applies
        "backbone_revision": None,
        "allow_licence": True,
        "mount_path": "/models/g",
    }
    assert "staged gliner" in capsys.readouterr().out


def test_cli_stage_reports_staging_errors(monkeypatch, tmp_path, capsys):
    def refuse(dest, **kwargs):
        raise im.LicenceError("no")

    monkeypatch.setattr(ims, "stage_gliner", refuse)
    assert im.main(["stage", "gliner", "--dest", str(tmp_path)]) == 2
    assert "error: no" in capsys.readouterr().err


def test_a_tokenizer_that_cannot_convert_is_a_named_staging_error(monkeypatch, tmp_path, capsys):
    """Without protobuf, transformers fell back to the TikToken loader and
    raised a raw ValueError ('Error parsing line') out of the CLI."""
    from types import SimpleNamespace

    def refuse(*_a, **_k):
        raise ValueError("Error parsing line b'\\x0e' in spm.model")

    fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=refuse))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(im.StagingError, match="protobuf"):
        ims._save_tokenizer(tmp_path / "backbone", tmp_path)

    def stage(dest, **_kw):
        ims._save_tokenizer(Path(dest) / "backbone", Path(dest))

    monkeypatch.setattr(ims, "stage_gliner", stage)
    assert im.main(["stage", "gliner", "--dest", str(tmp_path)]) == 2
    assert "error: the backbone tokenizer" in capsys.readouterr().err


def test_gliner_installs_come_with_protobuf():
    """umrs[kg-ingest] carries protobuf next to GLiNER (DeBERTa-v3 spm.model)."""
    pytest.importorskip("gliner")
    import google.protobuf  # noqa: F401 — the import is the check


SERVER_IMAGE = Path(__file__).resolve().parents[1] / "docker" / "server-image"


@pytest.mark.parametrize("lock", ["requirements.lock.txt", "requirements-cpu.lock.txt"])
def test_the_server_image_locks_pin_protobuf_for_gliner_staging(lock):
    """The image installs only its locks (--no-deps): without a protobuf pin,
    `ingest_models stage gliner` failed inside the image."""
    pins = (SERVER_IMAGE / lock).read_text().splitlines()
    assert any(line.startswith("gliner==") for line in pins), lock
    assert any(line.startswith("protobuf==") for line in pins), lock
    smoke = (SERVER_IMAGE / "smoke.py").read_text()
    assert '"google.protobuf"' in smoke


def test_cli_stage_docling(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(ims, "stage_docling", lambda dest: seen.append(dest) or Path(tmp_path))
    assert im.main(["stage", "docling", "--dest", "/x/docling"]) == 0
    assert seen == ["/x/docling"]


def test_cli_show(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(im.DOCLING_ARTIFACTS_ENV, str(tmp_path / "none"))
    monkeypatch.setenv(im.GLINER_MODEL_ENV, "off")
    monkeypatch.setenv(im.DOC_PARSER_ENV, "pypdf")
    monkeypatch.delenv(im.DOCMODELS_URL_ENV, raising=False)
    assert im.main(["show"]) == 0  # pypdf needs nothing staged
    out = capsys.readouterr().out
    assert f"{im.GLINER_MODEL_ENV}=off" in out and "(local only)" in out
    monkeypatch.delenv(im.GLINER_MODEL_ENV)
    assert im.main(["show"]) == 1  # on by default: the bundled dir is not staged here
    monkeypatch.setenv(im.DOC_PARSER_ENV, "docling")
    assert im.main(["show"]) == 1
    monkeypatch.setenv(im.DOC_PARSER_ENV, "pypdf")
    monkeypatch.setenv(im.GLINER_MODEL_ENV, str(tmp_path / "none"))
    assert im.main(["show"]) == 1
