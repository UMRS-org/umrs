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

    def info(self, repo, _revision):
        files = self.gliner_files if repo.startswith("urchade/") else ("config.json", "spm.model")
        licence = self.licences.get(
            repo, ("apache-2.0",) if repo.startswith("urchade/") else ("mit",)
        )
        return im.RepoInfo(sha=f"sha-{repo}", licences=licence, files=files)

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


def test_settings_defaults(monkeypatch):
    for name in (im.DOC_PARSER_ENV, im.DOCLING_ARTIFACTS_ENV, im.GLINER_MODEL_ENV):
        monkeypatch.delenv(name, raising=False)
    assert im.configured_doc_parser() == "pypdf"
    assert im.configured_docling_artifacts() == "/models/docling"
    assert im.configured_gliner_model() == ""


def test_settings_values(monkeypatch):
    monkeypatch.setenv(im.DOC_PARSER_ENV, " Docling ")
    monkeypatch.setenv(im.GLINER_MODEL_ENV, "  ")
    assert im.configured_doc_parser() == "docling"
    assert im.configured_gliner_model() == ""
    monkeypatch.setenv(im.DOC_PARSER_ENV, "marker")
    assert im.configured_doc_parser() == "pypdf"


def test_module_imports_no_heavy_library():
    code = (
        "import sys, umrs.knowledge.ingest_models; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'docling', 'gliner', 'huggingface_hub', 'transformers', 'torch'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ── Docling ─────────────────────────────────────────────────────────────────


def fake_download_models(
    output_dir=None,
    *,
    force=False,
    progress=False,
    with_layout=True,
    with_tableformer=True,
    with_code_formula=True,
    with_picture_classifier=True,
    with_rapidocr=True,
    with_easyocr=False,
):
    fake_download_models.calls.append(dict(locals()))
    if with_layout:
        (output_dir / im.DOCLING_SUBDIRS[0]).mkdir(parents=True)
        (output_dir / im.DOCLING_SUBDIRS[0] / "model.safetensors").write_bytes(b"l")
    if with_tableformer:
        tf = output_dir / im.DOCLING_SUBDIRS[1] / "model_artifacts" / "tableformer"
        tf.mkdir(parents=True)
        (tf / "tm_config.json").write_text("{}")


def test_docling_flags_only_layout_and_tableformer():
    flags = im.docling_download_flags(fake_download_models)
    assert flags == {
        "with_layout": True,
        "with_tableformer": True,
        "with_code_formula": False,
        "with_picture_classifier": False,
        "with_rapidocr": False,
        "with_easyocr": False,
    }


def test_docling_flags_refuse_an_api_without_layout():
    def old(output_dir=None, *, with_tableformer=True):
        return None

    with pytest.raises(im.StagingError, match="with_layout"):
        im.docling_download_flags(old)


def test_stage_docling_writes_marker_and_is_staged(tmp_path):
    fake_download_models.calls = []
    dest = tmp_path / "docling"
    assert not im.is_docling_staged(dest)
    assert im.stage_docling(dest, download=fake_download_models) == dest
    call = fake_download_models.calls[0]
    flags = {
        k: call[k]
        for k in (
            "progress",
            "with_layout",
            "with_tableformer",
            "with_rapidocr",
            "with_code_formula",
        )
    }
    assert call["output_dir"] == dest
    assert flags == {
        "progress": False,
        "with_layout": True,
        "with_tableformer": True,
        "with_rapidocr": False,
        "with_code_formula": False,
    }
    marker = json.loads((dest / im.MARKER).read_text())
    assert (marker["kind"], marker["subdirs"]) == ("docling", sorted(im.DOCLING_SUBDIRS))
    assert im.is_docling_staged(dest)


def test_stage_docling_fails_when_download_leaves_nothing(tmp_path):
    def empty(output_dir=None, *, with_layout=True, with_tableformer=True, progress=False):
        return None

    with pytest.raises(im.StagingError):
        im.stage_docling(tmp_path / "d", download=empty)


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
        im.backbone_repo(config, json.loads((path / im.MARKER).read_text()))
        == "microsoft/deberta-v3-base"
    )
    with pytest.raises(im.StagingError):
        im.backbone_repo(config, {})


def test_offline_config_keeps_a_shipped_encoder_config():
    config = {"model_name": "x/y", "encoder_config": {"model_type": "modernbert"}}
    out = im.offline_gliner_config(config, BACKBONE_CONFIG, "/m/backbone")
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
    assert im._licences_from("Apache-2.0", ()) == ("apache-2.0",)
    assert im._licences_from(["mit", "apache-2.0"], ()) == ("mit", "apache-2.0")
    assert im._licences_from(None, ["license:mit", "en"]) == ("mit",)
    assert im._licences_from(None, []) == ()


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_stage_gliner_passes_options(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_stage(dest, **kwargs):
        seen.update(kwargs, dest=dest)
        Path(dest).mkdir(parents=True)
        return Path(dest)

    monkeypatch.setattr(im, "stage_gliner", fake_stage)
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
        "revision": None,
        "allow_licence": True,
        "mount_path": "/models/g",
    }
    assert "staged gliner" in capsys.readouterr().out


def test_cli_stage_reports_staging_errors(monkeypatch, tmp_path, capsys):
    def refuse(dest, **kwargs):
        raise im.LicenceError("no")

    monkeypatch.setattr(im, "stage_gliner", refuse)
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
        im._save_tokenizer(tmp_path / "backbone", tmp_path)

    def stage(dest, **_kw):
        im._save_tokenizer(Path(dest) / "backbone", Path(dest))

    monkeypatch.setattr(im, "stage_gliner", stage)
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
    monkeypatch.setattr(im, "stage_docling", lambda dest: seen.append(dest) or Path(tmp_path))
    assert im.main(["stage", "docling", "--dest", "/x/docling"]) == 0
    assert seen == ["/x/docling"]


def test_cli_show(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(im.DOCLING_ARTIFACTS_ENV, str(tmp_path / "none"))
    monkeypatch.delenv(im.GLINER_MODEL_ENV, raising=False)
    monkeypatch.delenv(im.DOC_PARSER_ENV, raising=False)
    assert im.main(["show"]) == 0  # pypdf needs nothing staged
    assert "unset (off)" in capsys.readouterr().out
    monkeypatch.setenv(im.DOC_PARSER_ENV, "docling")
    assert im.main(["show"]) == 1
    monkeypatch.setenv(im.DOC_PARSER_ENV, "pypdf")
    monkeypatch.setenv(im.GLINER_MODEL_ENV, str(tmp_path / "none"))
    assert im.main(["show"]) == 1
