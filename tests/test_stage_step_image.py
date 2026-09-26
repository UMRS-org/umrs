"""Behaviour pins for ``deploy/_common/stage_step_image.sh`` (ADR-074 D-7).

The stager runs against a real temporary git checkout and a fake ``podman`` /
``uv`` pair whose output lives in files: ``podman save`` hands back a
docker-archive built here, so the layer scan, the id check, the sidecars, the
pruning and the refusals are asserted without a registry or a build. The
real build and a hardened run of the saved image are the manual receipt
recorded in the rccs step-lane spec (§9, §11).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

UMRS_ROOT = Path(__file__).resolve().parents[1]
STAGER = UMRS_ROOT / "deploy" / "_common" / "stage_step_image.sh"
CONTAINERFILE = UMRS_ROOT / "docker" / "Containerfile.step"
REQUIREMENTS = UMRS_ROOT / "docker" / "step-requirements.txt"

FAKE_PODMAN = r"""#!/usr/bin/env bash
state="${FAKE_STATE:?}"
printf '%s\n' "$*" >> "$state/podman.log"
while [[ "$1" == --root || "$1" == --runroot ]]; do shift 2; done
case "$1" in
  pull) cat "$state/base_id" ;;
  image)
    case "$4" in
      *RootFS.Layers*) cat "$state/base_layers" ;;
      *Architecture*) cat "$state/labels" ;;
      *.Id*) cat "$state/image_id" ;;
    esac ;;
  build) (cd "${@: -1}" && find . -printf '%m %p\n' | sort) > "$state/build_ctx" ;;
  save)
    while [[ $# -gt 0 ]]; do [[ "$1" == -o ]] && out="$2"; shift; done
    cp "$state/archive.tar" "$out" ;;
  unshare) shift; "$@" ;;
esac
"""

FAKE_UV = r"""#!/usr/bin/env bash
state="${FAKE_STATE:?}"
printf '%s\n' "$*" >> "$state/uv.log"
args=("$@")
for i in "${!args[@]}"; do
  case "${args[$i]}" in
    --out-dir) out="${args[$((i + 1))]}" ;;
    --target) target="${args[$((i + 1))]}" ;;
  esac
done
if [[ "$1" == build ]]; then
  (cd "${@: -1}" && find . -type f | sort) > "$state/wheel_src"
  : > "$out/umrs-0.2.0rc1-py3-none-any.whl"
  exit 0
fi
mkdir -p "$target/bin"
if [[ " $* " == *" --no-deps "* ]]; then
  info="$target/umrs-0.2.0rc1.dist-info"
  mkdir -p "$info" "$target/umrs"
  : > "$target/umrs/__init__.py"
  echo '{"url": "file:///var/tmp/x/wheel"}' > "$info/direct_url.json"
  echo '{}' > "$info/uv_cache.json"
  printf '%s\n' 'umrs/__init__.py,,' \
    'umrs-0.2.0rc1.dist-info/direct_url.json,sha256=x,2' \
    'umrs-0.2.0rc1.dist-info/uv_cache.json,sha256=y,2' \
    'umrs-0.2.0rc1.dist-info/RECORD,,' > "$info/RECORD"
  : > "$target/bin/umrs"
else
  mkdir -p "$target/numpy"
  : > "$target/numpy/__init__.py"
  : > "$target/bin/f2py"
fi
"""


def _layer(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as layer:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 1
            layer.addfile(info, io.BytesIO(b"x"))
    return buf.getvalue()


def write_docker_archive(
    path: Path,
    base: list[str],
    added: list[str],
    *,
    arch: str = "amd64",
    labels: dict[str, str] | None = None,
    tag: str = "localhost/umrs-step:x",
) -> tuple[str, list[str]]:
    """A docker-archive like ``podman save`` writes; returns (image id, diff ids)."""
    blobs = [_layer(base), _layer(added)]
    diff_ids = ["sha256:" + hashlib.sha256(b).hexdigest() for b in blobs]
    config = json.dumps(
        {
            "architecture": arch,
            "config": {"Labels": labels or {}},
            "rootfs": {"type": "layers", "diff_ids": diff_ids},
        }
    )
    image_id = hashlib.sha256(config.encode()).hexdigest()
    entries = {f"{image_id}.json": config.encode()}
    entries.update({f"{d[7:]}.tar": b for d, b in zip(diff_ids, blobs, strict=True)})
    manifest = [
        {
            "Config": f"{image_id}.json",
            "RepoTags": [tag],
            "Layers": [f"{d[7:]}.tar" for d in diff_ids],
        }
    ]
    entries["manifest.json"] = json.dumps(manifest).encode()
    with tarfile.open(path, "w") as out:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            out.addfile(info, io.BytesIO(data))
    return image_id, diff_ids


def _git(checkout: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


class Lane:
    """A checkout, an archive dir, fake tools and the state they read."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.checkout = tmp_path / "checkout"
        self.archive_dir = tmp_path / "archives"
        self.state = tmp_path / "state"
        self.bin = tmp_path / "bin"
        for d in (self.checkout / "src" / "umrs", self.checkout / "docker", self.state, self.bin):
            d.mkdir(parents=True)
        (self.checkout / "src" / "pyproject.toml").write_text("[project]\nname = 'umrs'\n")
        (self.checkout / "src" / "umrs" / "__init__.py").write_text("")
        shutil.copy(CONTAINERFILE, self.checkout / "docker")
        shutil.copy(REQUIREMENTS, self.checkout / "docker")
        _git(self.checkout, "init", "-q")
        _git(self.checkout, "add", "-A")
        _git(self.checkout, "commit", "-q", "-m", "pin")
        self.sha = _git(self.checkout, "rev-parse", "HEAD")
        self.epoch = _git(self.checkout, "log", "-1", "--format=%ct")
        self.archive = self.archive_dir / f"umrs-step-{self.sha[:12]}-x86_64.tar"
        for name, body in (("podman", FAKE_PODMAN), ("uv", FAKE_UV)):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        self.image(added=["usr/local/lib/python3.11/site-packages/umrs/__init__.py"])

    def archive_meta(self) -> dict:
        """What the stager's own build puts in an archive's config and manifest."""
        return {
            "labels": {
                "org.umrs.step-contract": "1",
                "org.opencontainers.image.revision": self.sha,
            },
            "tag": f"localhost/umrs-step:{self.sha[:12]}",
        }

    def image(self, added: list[str], base: list[str] | None = None) -> None:
        base = base if base is not None else ["etc/ssl/certs/ca.pem"]
        self.image_id, diff_ids = write_docker_archive(
            self.state / "archive.tar", base, added, **self.archive_meta()
        )
        (self.state / "base_id").write_text("baseid\n")
        (self.state / "base_layers").write_text(json.dumps(diff_ids[:1]) + "\n")
        (self.state / "labels").write_text(f"amd64 1 {self.sha}\n")
        (self.state / "image_id").write_text(self.image_id + "\n")

    def run(self, *extra: str, arch: str = "x86_64") -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "FAKE_STATE": str(self.state),
        }
        args = [
            "bash",
            str(STAGER),
            "--checkout",
            str(self.checkout),
            "--archive-dir",
            str(self.archive_dir),
        ]
        if arch:
            args += ["--arch", arch]
        return subprocess.run([*args, *extra], capture_output=True, text=True, env=env, check=False)

    def log(self, tool: str) -> str:
        path = self.state / f"{tool}.log"
        return path.read_text() if path.exists() else ""


def result(proc: subprocess.CompletedProcess[str]) -> dict:
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 1, proc.stdout + proc.stderr
    return json.loads(lines[0])


@pytest.fixture
def lane(tmp_path: Path) -> Lane:
    return Lane(tmp_path)


class TestRefusals:
    @pytest.mark.parametrize(
        ("args", "arch"),
        [((), "sparc"), (("--arch",), ""), (("--bogus",), "x86_64")],
    )
    def test_usage_errors_exit_2_with_a_failure_line(self, lane, args, arch):
        proc = lane.run(*args, arch=arch)
        assert proc.returncode == 2
        assert result(proc)["status"] == "failed"
        assert lane.log("podman") == ""

    def test_failure_line_stays_one_json_line_with_control_characters(self, lane):
        proc = lane.run(arch="x86\n64\t\x1b")
        assert proc.returncode == 2
        assert "x86 64" in result(proc)["reason"]

    def test_unreadable_authfile_is_a_usage_error(self, lane):
        proc = lane.run("--authfile", str(lane.tmp / "missing.json"))
        assert proc.returncode == 2

    def test_tracked_changes_refuse_the_build(self, lane):
        (lane.checkout / "src" / "umrs" / "__init__.py").write_text("changed = 1\n")
        proc = lane.run()
        assert proc.returncode == 3
        assert "tracked changes" in result(proc)["reason"]
        assert lane.log("uv") == "" and lane.log("podman") == ""

    def test_label_mismatch_refuses_the_image(self, lane):
        (lane.state / "labels").write_text("amd64 1 0000000000000000000000000000000000000000\n")
        proc = lane.run()
        assert proc.returncode == 1
        assert not lane.archive_dir.exists() or not any(lane.archive_dir.iterdir())

    def test_archive_id_must_equal_the_built_image_id(self, lane):
        (lane.state / "image_id").write_text("f" * 64 + "\n")
        proc = lane.run()
        assert proc.returncode == 1
        assert not lane.archive.exists()

    def test_image_must_start_with_the_base_layers(self, lane):
        (lane.state / "base_layers").write_text(json.dumps(["sha256:" + "0" * 64]) + "\n")
        proc = lane.run()
        assert proc.returncode == 1
        assert not lane.archive.exists()


class TestLayerScan:
    @pytest.mark.parametrize(
        "name",
        [
            "opt/app/env.0f1e2d3c",
            "opt/app/demo-llm.env",
            "opt/app/.env",
            "root/.s3cfg",
            "opt/app/s3cfg",
            "opt/app/s3cfg_user1",
            "home/u/.ssh/id_ed25519",
            "etc/tls/server.pem",
        ],
    )
    def test_forbidden_name_in_an_added_layer_refuses(self, lane, name):
        lane.image(added=["usr/local/lib/python3.11/site-packages/umrs/__init__.py", name])
        proc = lane.run()
        assert proc.returncode == 4, proc.stderr
        assert name in proc.stderr
        assert result(proc)["status"] == "failed"
        assert list(lane.archive_dir.iterdir()) == []

    def test_base_layers_and_lookalike_modules_pass(self, lane):
        lane.image(
            base=["etc/ssl/certs/ca-certificates.pem", "root/.env"],
            added=["usr/local/lib/python3.11/site-packages/umrs/shared/storage/user_s3cfg.py"],
        )
        proc = lane.run()
        assert proc.returncode == 0, proc.stderr
        assert result(proc)["scan"]["hits"] == 0


class TestBuild:
    def test_writes_archive_with_sidecars_and_reports_it(self, lane):
        proc = lane.run()
        assert proc.returncode == 0, proc.stderr
        out = result(proc)
        digest = hashlib.sha256(lane.archive.read_bytes()).hexdigest()
        assert out["status"] == "built"
        assert out["image"] == f"localhost/umrs-step:{lane.sha[:12]}"
        assert (out["image_id"], out["sha256"], out["revision"]) == (
            lane.image_id,
            digest,
            lane.sha,
        )
        assert (
            lane.archive_dir / (lane.archive.name + ".sha256")
        ).read_text() == f"{digest}  {lane.archive.name}\n"
        assert (lane.archive_dir / (lane.archive.name + ".id")).read_text() == lane.image_id + "\n"
        assert sorted(p.name for p in lane.archive_dir.iterdir()) == sorted(
            [lane.archive.name, lane.archive.name + ".sha256", lane.archive.name + ".id"]
        )

    @staticmethod
    def _arm_build(lane: Lane) -> subprocess.CompletedProcess[str]:
        (lane.state / "labels").write_text(f"arm64 1 {lane.sha}\n")
        proc = lane.run("--authfile", str(lane.state / "base_id"), arch="aarch64")
        assert proc.returncode == 0, proc.stderr
        return proc

    def test_throwaway_store_is_removed_afterwards(self, lane):
        proc = self._arm_build(lane)
        root = re.search(r"--root (/var/tmp/umrs-step-stage\.\w+)/root", lane.log("podman"))
        assert not Path(root.group(1)).exists()
        assert result(proc)["archive"].endswith(f"-{lane.sha[:12]}-aarch64.tar")

    def test_podman_builds_the_target_arch_from_the_pinned_commit(self, lane):
        self._arm_build(lane)
        podman = lane.log("podman")
        assert f"--platform linux/arm64 --authfile {lane.state / 'base_id'}" in podman
        assert f"--pull=never --platform linux/arm64 --timestamp {lane.epoch}" in podman
        assert f"UMRS_REVISION={lane.sha}" in podman

    def test_uv_installs_hash_checked_binary_wheels_for_the_target_arch(self, lane):
        self._arm_build(lane)
        uv = lane.log("uv")
        assert uv.count("--python-platform aarch64-manylinux_2_28") == 2
        assert "--only-binary :all: --require-hashes -r" in uv
        assert "--no-deps" in uv

    def test_site_packages_is_readable_by_the_job_user(self, lane):
        assert lane.run().returncode == 0
        ctx = (lane.state / "build_ctx").read_text().splitlines()
        site = [line.split()[0] for line in ctx if "./site-packages/" in line]
        unreadable = [mode for mode in site if int(mode, 8) & 0o044 != 0o044]
        assert site
        assert unreadable == []

    def test_site_packages_carries_no_build_host_state(self, lane):
        assert lane.run().returncode == 0
        ctx = (lane.state / "build_ctx").read_text()
        assert "./Containerfile.step" in ctx
        leaked = [
            n for n in ("./site-packages/bin", "direct_url.json", "uv_cache.json") if n in ctx
        ]
        assert leaked == []

    def test_record_drops_the_removed_install_metadata(self, lane):
        record = lane.state / "record_seen"
        (lane.bin / "podman").write_text(
            FAKE_PODMAN.replace(
                "build) (cd",
                f'build) cat "${{@: -1}}"/site-packages/umrs-*.dist-info/RECORD > {record}; (cd',
            )
        )
        assert lane.run().returncode == 0
        assert record.read_text().splitlines() == [
            "umrs/__init__.py,,",
            "umrs-0.2.0rc1.dist-info/RECORD,,",
        ]

    def test_untracked_files_never_reach_the_wheel(self, lane):
        (lane.checkout / "src" / "umrs" / "env.0f1e2d3c").write_text("LITELLM_MASTER_KEY=x\n")
        assert lane.run().returncode == 0
        wheel_src = (lane.state / "wheel_src").read_text()
        assert "./umrs/__init__.py" in wheel_src
        assert "env.0f1e2d3c" not in wheel_src


class TestIfMissingAndPruning:
    def test_present_archive_is_verified_and_not_rebuilt(self, lane):
        assert lane.run().returncode == 0
        (lane.state / "podman.log").unlink()
        proc = lane.run("--if-missing")
        assert proc.returncode == 0
        assert result(proc)["status"] == "present"
        assert lane.log("podman") == ""

    def test_a_re_pinned_present_archive_becomes_the_newest(self, lane):
        """Pruning goes by mtime: re-pinning an older archive must mark it in
        use, or the next build prunes the image the server still runs."""
        assert lane.run().returncode == 0
        old = 1_000_000_000
        for path in (lane.archive, Path(f"{lane.archive}.sha256"), Path(f"{lane.archive}.id")):
            os.utime(path, (old, old))
        assert lane.run("--if-missing").returncode == 0
        assert lane.archive.stat().st_mtime > old + 1

    def test_corrupt_archive_is_rebuilt_under_if_missing(self, lane):
        assert lane.run().returncode == 0
        lane.archive.write_bytes(b"truncated")
        proc = lane.run("--if-missing")
        assert proc.returncode == 0
        assert result(proc)["status"] == "built"
        assert hashlib.sha256(lane.archive.read_bytes()).hexdigest() == result(proc)["sha256"]

    # A job loads the archive, then requires the loaded image to carry the
    # .id, the contract label and the local name: an archive whose sha256
    # still matches but that fails any of these would fail every job.
    @pytest.mark.parametrize(
        "tamper",
        [
            {"id": "f" * 64},
            {"id": ""},
            {"id": "prefixed"},
            {"arch": "arm64"},
            {"labels": {"org.umrs.step-contract": "2"}},
            {"labels": {"org.umrs.step-contract": "1"}},
            {"tag": "localhost/umrs-step:000000000000"},
        ],
    )
    def test_present_archive_jobs_would_refuse_is_rebuilt(self, lane, tamper):
        assert lane.run().returncode == 0
        good = lane.archive.read_bytes()
        sidecar_id = Path(f"{lane.archive}.id")
        if "id" in tamper:
            text = tamper["id"]
            sidecar_id.write_text(
                ("sha256:" + lane.image_id if text == "prefixed" else text) + "\n"
            )
        else:
            meta = {**lane.archive_meta(), **tamper}
            added = ["usr/local/lib/python3.11/site-packages/umrs/__init__.py"]
            image_id, _ = write_docker_archive(
                lane.archive, ["etc/ssl/certs/ca.pem"], added, **meta
            )
            digest = hashlib.sha256(lane.archive.read_bytes()).hexdigest()
            Path(f"{lane.archive}.sha256").write_text(f"{digest}  {lane.archive.name}\n")
            sidecar_id.write_text(image_id + "\n")
        proc = lane.run("--if-missing")
        assert proc.returncode == 0, proc.stderr
        assert result(proc)["status"] == "built"
        assert lane.archive.read_bytes() == good
        assert sidecar_id.read_text() == lane.image_id + "\n"

    def test_keeps_the_newest_three_per_arch(self, lane):
        lane.archive_dir.mkdir()
        old = []
        for age, tag in enumerate(
            ("aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa3", "aaaaaaaaaaa4"), start=1
        ):
            path = lane.archive_dir / f"umrs-step-{tag}-x86_64.tar"
            for p in (path, Path(f"{path}.sha256"), Path(f"{path}.id")):
                p.write_text("x")
                os.utime(p, (1_000_000 - age * 100, 1_000_000 - age * 100))
            old.append(path.name)
        untouched = ["umrs-step-aaaaaaaaaaa9-aarch64.tar", "r4-mcp-latest-arm64.tar"]
        for name in untouched:
            (lane.archive_dir / name).write_text("x")
        proc = lane.run()
        assert proc.returncode == 0, proc.stderr
        names = {p.name for p in lane.archive_dir.iterdir()}
        assert {lane.archive.name, old[0], old[1], *untouched} <= names
        assert not {old[2], old[3], f"{old[3]}.sha256", f"{old[3]}.id"} & names
        assert sorted(result(proc)["pruned"]) == sorted(old[2:])


class TestRecipeFiles:
    @staticmethod
    def _containerfile_code() -> list[str]:
        lines = CONTAINERFILE.read_text().splitlines()
        return [line for line in lines if line and not line.startswith("#")]

    def test_containerfile_base_is_digest_pinned(self):
        assert re.match(r"FROM \S+:3\.11-slim@sha256:[0-9a-f]{64}$", self._containerfile_code()[0])

    def test_containerfile_is_copy_only(self):
        instructions = {line.split()[0] for line in self._containerfile_code() if line[0].isupper()}
        assert instructions & {"RUN", "ADD", "ENTRYPOINT"} == set()

    def test_containerfile_labels_and_default_user(self):
        text = CONTAINERFILE.read_text()
        assert 'org.umrs.step-contract="1"' in text
        assert 'org.opencontainers.image.revision="${UMRS_REVISION}"' in text
        assert "USER 65534:65534" in self._containerfile_code()

    def test_requirements_are_hash_pinned_core_tier_without_boto3(self):
        text = REQUIREMENTS.read_text()
        pins = dict(re.findall(r"^([a-z0-9-]+)==(\S+) \\$", text, flags=re.M))
        assert {
            "numpy",
            "pandas",
            "pyyaml",
            "pillow",
            "opencv-python-headless",
            "jsonschema",
        } <= set(pins)
        assert not {"boto3", "botocore", "s3transfer"} & set(pins)
        blocks = re.split(r"^(?=[a-z0-9-]+==)", text, flags=re.M)[1:]
        assert len(blocks) == len(pins) and all("--hash=sha256:" in b for b in blocks)

    def test_stager_syntax(self):
        proc = subprocess.run(
            ["bash", "-n", str(STAGER)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_stager_shellcheck_clean(self):
        proc = subprocess.run(
            ["shellcheck", "-s", "bash", str(STAGER)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stdout
