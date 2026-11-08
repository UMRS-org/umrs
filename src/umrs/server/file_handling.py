"""Server-side file operations: user config persistence, workflow upload."""

import json
import logging
import os
import re
import shutil
import struct
import uuid
import zipfile
from dataclasses import dataclass

from umrs.server.workflow_versions import (
    invalid_version_message,
    is_semver,
    name_suffix_warning,
    tree_digest,
    tree_digest_or_none,
    version_dir_name,
)
from umrs.shared.auth import sanitize_username, validate_config_identity
from umrs.shared.config import (
    get_default_tmp_subdir,
    get_resolved_umrs_subdir,
    get_users_default_cfg_subdir,
)
from umrs.shared.json_config import load_client_cfg_file

logger = logging.getLogger(__name__)


#: Upload notice when the tree carries a Dockerfile (ADR-074 D-7).
DOCKERFILE_IGNORED_NOTICE = (
    "Dockerfile ignored: the server no longer builds images from uploaded "
    "workflows (ADR-074 D-7); steps run in the platform step image or one on "
    "the site's allowed_images list"
)


def _dockerfile_notice(dest_dir: str, workflow_name: str) -> str | None:
    """The upload's notice when it carries a Dockerfile, which is never built:
    a build runs the upload's own instructions on the server host."""
    if not os.path.isfile(os.path.join(dest_dir, "Dockerfile")):
        return None
    logger.warning("workflow upload %r: %s", workflow_name, DOCKERFILE_IGNORED_NOTICE)
    return DOCKERFILE_IGNORED_NOTICE


def ensure_user_file_in_conf(conf_path, username, canonical_user_path):
    """Ensure the server conf JSON lists exactly one config entry for *username*."""
    canonical_user_path = os.path.expanduser(canonical_user_path)

    try:
        with open(conf_path, encoding="utf-8") as f:
            conf = json.load(f)
    except Exception as e:
        logger.info(f"Warning: failed to read conf '{conf_path}': {e}")
        return False

    users = conf.get("users", [])

    # Build a new list that keeps only one entry for this username (the canonical path)
    new_users = []
    removed = []
    for p in users:
        p = os.path.expanduser(p)
        if p == canonical_user_path:
            # keep canonical path if already present
            if p not in new_users:
                new_users.append(p)
            continue
        # Try to read file and inspect username; if mismatch or file missing, keep as-is
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            u = data.get("username")
        except Exception:
            u = None
        if u == username:
            # Another config for the same user; drop it
            removed.append(p)
            continue
        # Different user or unreadable -> keep entry
        if p not in new_users:
            new_users.append(p)

    # Ensure canonical path present
    if canonical_user_path not in new_users:
        new_users.append(canonical_user_path)

    if removed:
        logger.info(
            f"Updated server conf: removed {len(removed)} stale user config(s) "
            f"for '{username}': {removed}"
        )

    # change absolute path to '~' if possible
    home = os.path.expanduser("~")
    new_users = [path.replace(home, "~", 1) for path in new_users]

    conf["users"] = new_users
    try:
        with open(conf_path, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=2)
        logger.info(
            f"Updated server conf: ensured unique config for "
            f"'{username}' -> {canonical_user_path}"
        )
    except Exception as e:
        logger.info(f"Warning: failed to update conf '{conf_path}': {e}")
        return False

    # Best-effort cleanup: remove pruned files under user_cfg_path only
    user_cfg_path = os.path.join(
        os.path.dirname(conf_path), get_users_default_cfg_subdir()
    )
    for stale in removed:
        try:
            if (
                stale.startswith(user_cfg_path)
                and stale != canonical_user_path
                and os.path.isfile(stale)
            ):
                os.remove(stale)
                logger.info(f"Deleted stale user config file: {stale}")
        except Exception as e:
            logger.info(f"Warning: failed to delete stale user config '{stale}': {e}")
    return True


#: An uploaded workflow's name is one directory under its owner's extracted
#: tree: an absolute name or ``..`` put the upload's files wherever the server
#: may write (its own modules, another owner's workflow tree).
WORKFLOW_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_WORKFLOW_JSON_BYTES = 1024 * 1024


class _UploadRefused(Exception):
    """Ends an upload with this message (*code*: a typed refusal, ADR-113)."""

    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code


def workflow_name_refusal(workflow_name: object) -> str | None:
    """Why *workflow_name* cannot name an uploaded workflow, or None."""
    if isinstance(workflow_name, str) and WORKFLOW_NAME_RE.fullmatch(workflow_name):
        return None
    return (
        f"Invalid workflow_name {str(workflow_name)[:80]!r}: 1-128 characters of "
        "[A-Za-z0-9._-], starting with a letter or digit"
    )


def _within(root: str, path: str) -> bool:
    """*path* is *root* or below it (both real paths); unlike a prefix test,
    ``<root>2`` is not below ``<root>``."""
    return os.path.commonpath([root, path]) == root


def _precheck_upload(zip_path: str, workflow_name: object) -> None:
    refusal = workflow_name_refusal(workflow_name)
    if refusal:
        raise _UploadRefused(refusal)
    if not os.path.isfile(zip_path):
        raise _UploadRefused(f"Uploaded file not found: {zip_path}")
    size = os.path.getsize(zip_path)
    if size > MAX_UPLOAD_BYTES:
        raise _UploadRefused(
            f"Zip file too large ({size} bytes, max {MAX_UPLOAD_BYTES})"
        )
    if not zipfile.is_zipfile(zip_path):
        raise _UploadRefused(f"Uploaded file is not a valid zip archive: {zip_path}")


#: ``(filename, extra, comment)`` lengths in a central directory header.
_CD_NAME_LENGTHS = struct.Struct("<3H")
_CD_NAME_LENGTHS_AT = 28


def _directory_records(data: bytes, cap: int) -> int:
    """Central directory records at the start of *data*, walked as
    ``ZipFile`` walks them (to the first bad magic number); stops past
    *cap*."""
    count = pos = 0
    while (
        count <= cap
        and len(data) - pos >= zipfile.sizeCentralDir
        and data[pos : pos + 4] == zipfile.stringCentralDir
    ):
        count += 1
        lengths = _CD_NAME_LENGTHS.unpack_from(data, pos + _CD_NAME_LENGTHS_AT)
        pos += zipfile.sizeCentralDir + sum(lengths)
    return count


class _CountedDirectory:
    """The upload as ``ZipFile`` opens it. ``ZipFile`` reads the central
    directory in one ``read`` after seeking to it (every CPython from 3.8 to
    3.14; where it is depends on the version) and then builds one object per
    record, whatever the end record's entry count says: a 500 MB zip of
    minimal entries was ~6M objects before any cap applied. The records are
    counted in that read, and more than *cap* refused before one object is
    built. Disarmed once the archive is open: member reads are not
    directory reads."""

    def __init__(self, raw, cap: int) -> None:
        self._raw, self._cap, self.armed = raw, cap, True

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        if self.armed and _directory_records(data, self._cap) > self._cap:
            raise _UploadRefused(f"Zip holds more than {self._cap} entries")
        return data

    def __getattr__(self, name: str):
        return getattr(self._raw, name)


def workflows_root(client_id: str) -> str:
    """The owner's extracted-workflows directory (real, absolute path):
    ``<UMRS_WORKFLOW_STATE_DIR>/extracted/<owner>`` on deployed servers
    (a durable bind-mounted dir), else the umrs tmp tree."""
    state_dir = os.environ.get("UMRS_WORKFLOW_STATE_DIR", "").strip()
    if state_dir:
        workflows_dir: str = os.path.join(state_dir, "extracted", client_id)
    else:
        workflows_dir = get_resolved_umrs_subdir(
            get_default_tmp_subdir(), "user_workflows", client_id
        )
    return os.path.realpath(workflows_dir)


def _upload_dest(client_id: str, workflow_name: str) -> str:
    """The workflow's NAME directory, an ABSOLUTE path so that the stored
    ``source_path`` resolves regardless of the reader's working directory
    (the durable-job backend). Each registered version lives in its own
    subdirectory, ``<name>/<version>/`` (ADR-113 D-2)."""
    root = workflows_root(client_id)
    dest_dir = os.path.realpath(os.path.join(root, workflow_name))
    if dest_dir == root or not _within(root, dest_dir):
        raise _UploadRefused(
            f"workflow_name {workflow_name!r} leaves the upload directory"
        )
    return dest_dir


_S_IFMT = 0o170000
_ZIP_MEMBER_TYPES = (0, 0o100000, 0o040000)  # unset (DOS), regular file, directory


def _check_zip_size(zf) -> None:
    """The extracted tree fits the source caps the step lane stages (spec §5),
    checked before anything is written: a small zip of compressible members
    expands to whatever it declares in the server's workflow state dir,
    which every server on the host shares. ``zipfile`` never writes more
    than a member's declared ``file_size``."""
    from umrs.server.step_staging import DEFAULT_CAPS

    infos = zf.infolist()
    files = [info for info in infos if not info.is_dir()]
    total = sum(info.file_size for info in files)
    if (
        len(infos) > 2 * DEFAULT_CAPS.max_src_files
        or len(files) > DEFAULT_CAPS.max_src_files
    ):
        raise _UploadRefused(
            f"Zip holds {len(files)} files in {len(infos)} entries (max "
            f"{DEFAULT_CAPS.max_src_files} files)"
        )
    if total > DEFAULT_CAPS.max_src_bytes:
        raise _UploadRefused(
            f"Zip expands to {total} bytes (max {DEFAULT_CAPS.max_src_bytes})"
        )
    for info in infos:
        if ((info.external_attr >> 16) & _S_IFMT) not in _ZIP_MEMBER_TYPES:
            raise _UploadRefused(
                f"Zip member {info.filename!r} is not a regular file or directory"
            )


def _package_check(raw: bytes) -> tuple[list[str], str]:
    """The upload contract (r4_contracts.workflow_package, 2026-10-02): a
    description and an explicit flow graph (every step's ``depends_on``)
    are mandatory, so an error refuses the upload, every problem named at
    once; ``version`` must be a semantic version (ADR-113 D-1, refused by
    name as ``workflows.invalid_version``). Returns the contract's warnings
    as advice lines for the upload notice, and the version."""
    from r4_contracts.workflow_package import (
        errors_of,
        format_problems,
        warnings_of,
        workflow_package_problems,
    )

    from umrs.pipeline_sdk.workflow import WorkflowDefinition

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _UploadRefused(f"Invalid workflow.json: {e}") from e
    problems = workflow_package_problems(data)
    errors = errors_of(problems)
    if errors:
        raise _UploadRefused(f"Invalid workflow.json: {format_problems(errors)}")
    try:
        WorkflowDefinition.from_dict(data)
    except (KeyError, ValueError) as e:
        raise _UploadRefused(f"Invalid workflow.json: {e}") from e
    version = data.get("version") if isinstance(data, dict) else None
    if not is_semver(version):
        raise _UploadRefused(
            invalid_version_message(version), "workflows.invalid_version"
        )
    return [str(p) for p in warnings_of(problems)], str(version)


def _workflow_json_member(names: list[str]) -> str:
    """The bundle's workflow.json: the root one, else the first (sorted) one
    exactly one level deep. A deeper one never defines the bundle (before,
    whichever came first in the archive was checked, and another used)."""
    if "workflow.json" in names:
        return "workflow.json"
    nested = sorted(
        n for n in names if n.count("/") == 1 and n.endswith("/workflow.json")
    )
    if not nested:
        raise _UploadRefused(
            "Zip archive does not contain a workflow.json at its root or one level deep"
        )
    return nested[0]


def _extract_checked(zf, dest_dir: str) -> tuple[list[str], str, str]:
    """Extract after every member is checked to stay inside *dest_dir*, the
    archive fits the source caps and its ``workflow.json`` (root or one level
    deep) meets the upload contract and parses. Returns the contract's
    warnings, the workflow.json member and the bundle's version."""

    _check_zip_size(zf)
    for member in zf.namelist():
        if not _within(dest_dir, os.path.realpath(os.path.join(dest_dir, member))):
            raise _UploadRefused(f"Zip contains unsafe path: {member}")
    for name in zf.namelist():
        # Parsed whole into server memory: a small zip of a compressible
        # one expanded to max_src_bytes of JSON objects (~20 GiB of Python).
        declared = zf.getinfo(name).file_size
        if os.path.basename(name) == "workflow.json" and (
            declared > MAX_WORKFLOW_JSON_BYTES
        ):
            raise _UploadRefused(
                f"{name} is {declared} bytes (max {MAX_WORKFLOW_JSON_BYTES})"
            )
    member = _workflow_json_member(zf.namelist())
    warnings, version = _package_check(zf.read(member))
    os.makedirs(dest_dir, exist_ok=True)
    zf.extractall(dest_dir)
    return warnings, member, version


def _flatten_nested(dest_dir: str, member: str = "workflow.json") -> None:
    """A workflow.json one level deep (``myworkflow/workflow.json``): that
    directory's contents move up so *dest_dir* holds workflow.json. The
    directory is renamed aside first, so an item named like it moves too
    (``wf/wf/``), and an item also present at the root is refused, never
    merged."""
    if member == "workflow.json":
        return
    sub_dir = os.path.join(dest_dir, member.split("/", 1)[0])
    aside = os.path.join(dest_dir, f".flatten-{uuid.uuid4().hex}")
    os.rename(sub_dir, aside)
    for item in sorted(os.listdir(aside)):
        target = os.path.join(dest_dir, item)
        if os.path.lexists(target):
            raise _UploadRefused(
                f"Zip member {item!r} exists both at the root and next to {member}"
            )
        shutil.move(os.path.join(aside, item), target)
    os.rmdir(aside)


def _mark_uploaded(dest_dir: str, final_dir: str | None = None) -> None:
    """Rewrite workflow.json's source to the UPLOADED tree, which must lie
    inside *dest_dir*; the path written is where the tree will live
    (*final_dir*, the version directory it is renamed to)."""
    actual_wf = os.path.join(dest_dir, "workflow.json")
    if not os.path.isfile(actual_wf):
        return
    with open(actual_wf, encoding="utf-8") as fh:
        wf_data = json.load(fh)
    source_path = (wf_data.get("source") or {}).get("path") or "."
    # An absolute source path replaces dest_dir in the join.
    resolved_source = os.path.realpath(os.path.join(dest_dir, source_path))
    if not _within(dest_dir, resolved_source):
        raise _UploadRefused(
            f"Workflow source path escapes extracted directory: {source_path}"
        )
    if not os.path.isdir(resolved_source):
        raise _UploadRefused(
            f"Workflow source directory missing after extraction: {resolved_source}"
        )
    final_source = os.path.normpath(
        os.path.join(final_dir or dest_dir, os.path.relpath(resolved_source, dest_dir))
    )
    wf_data["source"] = {"type": "uploaded", "path": final_source}
    with open(actual_wf, "w", encoding="utf-8") as fh:
        json.dump(wf_data, fh, indent=2)


@dataclass(frozen=True)
class UploadOutcome:
    """One upload's result (ADR-113): where the version lives, its digest,
    and whether the same bundle was already registered (``unchanged``)."""

    ok: bool
    dest: str | None = None
    message: str | None = None
    code: str = ""
    version: str = ""
    source_sha256: str = ""
    unchanged: bool = False
    warnings: tuple[str, ...] = ()


def _staging_dir(name_dir: str) -> str:
    """An empty directory beside the owner's workflows (``.incoming`` never
    names a workflow: a name starts with a letter or digit)."""
    return os.path.join(os.path.dirname(name_dir), ".incoming", uuid.uuid4().hex)


def _extract_to(zip_path: str, staging: str) -> tuple[list[str], str]:
    """Checked extraction into the empty *staging* dir; (advice, version)."""
    from umrs.server.step_staging import DEFAULT_CAPS

    with open(zip_path, "rb") as raw:
        guarded = _CountedDirectory(raw, 2 * DEFAULT_CAPS.max_src_files)
        with zipfile.ZipFile(guarded) as zf:
            guarded.armed = False
            advice, member, version = _extract_checked(zf, staging)
    _flatten_nested(staging, member)
    return advice, version


def _same_version(final_dir: str, digest: str, version: str) -> None:
    """A registered version is immutable (ADR-113 D-2): the same bundle
    again is a no-op, other content is refused by name."""
    if tree_digest_or_none(final_dir) != digest:
        raise _UploadRefused(
            f"version {version} already exists with different content; "
            "bump `version` in workflow.json",
            "workflows.version_exists",
        )


def _place_version(staging: str, name_dir: str, version: str) -> tuple[str, str, bool]:
    """Move the staged tree to ``<name>/<version>/``: (dir, digest, unchanged)."""
    digest = tree_digest(staging)
    final_dir = os.path.join(name_dir, version_dir_name(version))
    if os.path.isdir(final_dir):
        _same_version(final_dir, digest, version)
        return final_dir, digest, True
    _mark_uploaded(staging, final_dir)
    os.makedirs(name_dir, exist_ok=True)
    try:
        os.rename(staging, final_dir)
    except OSError:
        if not os.path.isdir(final_dir):
            raise
        _same_version(final_dir, digest, version)  # a concurrent upload won
        return final_dir, digest, True
    return final_dir, digest, False


def _discard_upload(staging: str | None, zip_path: str) -> None:
    if staging and os.path.isdir(staging):
        shutil.rmtree(staging, ignore_errors=True)
    try:
        os.remove(zip_path)
    except Exception as e:
        logger.info(f"Warning: failed to clean up zip file '{zip_path}': {e}")


def _uploaded(
    client_id: str,
    workflow_name: str,
    version: str,
    placed: tuple[str, str, bool],
    advice: list[str],
) -> UploadOutcome:
    final_dir, digest, unchanged = placed
    logger.info(
        "Workflow '%s' v%s for user '%s' %s %s",
        workflow_name,
        version,
        client_id,
        "already registered at" if unchanged else "extracted to",
        final_dir,
    )
    warning = name_suffix_warning(workflow_name)
    notices = [_dockerfile_notice(final_dir, workflow_name), *advice, warning]
    return UploadOutcome(
        True,
        dest=final_dir,
        message="; ".join(n for n in notices if n) or None,
        version=version,
        source_sha256=digest,
        unchanged=unchanged,
        warnings=(warning,) if warning else (),
    )


def upload_workflow_bundle(
    client_id: str, zip_path: str, workflow_name: str
) -> UploadOutcome:
    """Validate an uploaded workflow zip and store it as one immutable
    version, ``<owner>/<name>/<version>/`` (ADR-113 D-2).

    The zip must contain a ``workflow.json`` at its root (or one level deep)
    that meets the upload contract (a description, every step's
    ``depends_on``, ``r4_contracts.workflow_package``), carries a SemVer
    ``version`` and parses as a :class:`WorkflowDefinition`; *workflow_name*
    must match :data:`WORKFLOW_NAME_RE`. The tree is extracted into an empty
    staging directory and renamed into place: a version is never written
    over. The same bundle again is ``unchanged``; other content under an
    existing version is refused (``workflows.version_exists``). The uploaded
    zip is always removed.
    """
    staging: str | None = None
    try:
        _precheck_upload(zip_path, workflow_name)
        name_dir = _upload_dest(client_id, workflow_name)
        staging = _staging_dir(name_dir)
        advice, version = _extract_to(zip_path, staging)
        placed = _place_version(staging, name_dir, version)
    except _UploadRefused as refused:
        return UploadOutcome(False, message=str(refused), code=refused.code)
    except zipfile.BadZipFile:
        return UploadOutcome(False, message=f"Corrupted zip archive: {zip_path}")
    except Exception as e:
        return UploadOutcome(False, message=f"Failed to extract workflow: {e}")
    finally:
        _discard_upload(staging, zip_path)
    return _uploaded(client_id, workflow_name, version, placed, advice)


def _set_identity(tree: str, old_root: str, workflow_name: str, version: str) -> None:
    """An adopted tree's workflow.json names its new identity, and an
    absolute ``source.path`` into the old tree becomes relative."""
    path = os.path.join(tree, "workflow.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    source = data.get("source")
    old = source.get("path") if isinstance(source, dict) else None
    if isinstance(old, str) and os.path.isabs(old):
        real_old = os.path.realpath(old)
        if _within(old_root, real_old):
            source["path"] = os.path.relpath(real_old, old_root)
    data["version"] = version
    if "name" in data:
        data["name"] = workflow_name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def adopt_version_tree(
    client_id: str, source_dir: str, workflow_name: str, version: str
) -> UploadOutcome:
    """Store a copy of a registered tree as (*workflow_name*, *version*) of
    the same owner (ADR-113 D-8 adopt): the same staging, immutability and
    digest rules as an upload. Symlinks are copied as links (never
    followed) and are not part of the digest."""
    staging: str | None = None
    try:
        name_dir = _upload_dest(client_id, workflow_name)
        staging = _staging_dir(name_dir)
        shutil.copytree(source_dir, staging, symlinks=True)
        _set_identity(staging, os.path.realpath(source_dir), workflow_name, version)
        placed = _place_version(staging, name_dir, version)
    except _UploadRefused as refused:
        return UploadOutcome(False, message=str(refused), code=refused.code)
    except (OSError, ValueError) as e:
        return UploadOutcome(False, message=f"Failed to adopt workflow: {e}")
    finally:
        if staging and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
    return _uploaded(client_id, workflow_name, version, placed, [])


def handle_upload_workflow(client_id, zip_path, workflow_name):
    """:func:`upload_workflow_bundle` as ``(success, dest_dir, message)``:
    the error on failure; on success the version directory and an optional
    notice for the uploader (a Dockerfile in the upload is ignored, never
    built — ADR-074 D-7; the contract's advice; a version-like name)."""
    outcome = upload_workflow_bundle(client_id, zip_path, workflow_name)
    return outcome.ok, outcome.dest, outcome.message


def persist_uploaded_user_config(
    tmp_path,
    configure_file,
    user_config_map,
    expected_username=None,
    *,
    allow_create=False,
):
    """Persist an uploaded user config JSON.

    When *allow_create* is ``False`` (the default) the function refuses to
    create entries for usernames that are not already present in
    *user_config_map*.  Only the admin-only ``ENROLL_USER`` path should
    pass ``allow_create=True``.

    Returns ``(success: bool, username: str | None, dest_path: str | None)``.
    """
    try:
        # Validate against client schema (strict) and obtain data
        data = load_client_cfg_file(tmp_path)
        username = data.get("username")
        if not username:
            logger.info(
                f"Warning: uploaded file '{tmp_path}' missing 'username' field; "
                f"ignoring"
            )
            return False, None, None

        # Sanitize username to prevent path-traversal attacks
        safe_username = sanitize_username(username)
        if safe_username != username:
            logger.warning(
                "Rejecting unsafe username: %r (sanitized: %r)",
                username,
                safe_username,
            )
            logger.info(
                f"Warning: uploaded file '{tmp_path}' contains unsafe "
                f"username '{username}'; rejecting"
            )
            return False, None, None

        # Rich identity validation (checks username match + flags admin claims)
        if expected_username:
            id_result = validate_config_identity(data, expected_username)
            if not id_result.valid:
                logger.info(
                    f"Warning: uploaded config identity validation failed: "
                    f"{'; '.join(id_result.errors)}; ignoring file"
                )
                return False, None, None
            for w in id_result.warnings:
                logger.warning("Config identity warning: %s", w)
    except SystemExit:
        # load_client_cfg_file may sys.exit() on validation error
        logger.info(f"Warning: uploaded file '{tmp_path}' failed validation; ignoring")
        return False, None, None
    except Exception as e:
        logger.info(f"Warning: cannot parse uploaded config '{tmp_path}': {e}")
        return False, None, None

    # look up existing users configuration path
    dest_path = user_config_map.get(username, None)

    if not dest_path:
        if not allow_create:
            logger.info(
                f"Rejected: user '{username}' is not enrolled on this server. "
                f"An admin must run 'admin user add' first."
            )
            return False, username, None

        # Destination path (canonical) — only reached when allow_create=True
        umrs_dir = os.path.dirname(configure_file)
        user_dir = os.path.join(umrs_dir, get_users_default_cfg_subdir())
        try:
            os.makedirs(user_dir, exist_ok=True)
        except Exception as e:
            logger.info(
                f"Warning: cannot ensure user files directory '{user_dir}': {e}"
            )
            return False, username, None
        dest_path = os.path.join(user_dir, f"{username}.json")

    dest_path = os.path.expanduser(dest_path)

    # Atomically replace the destination file
    try:
        with open(tmp_path, encoding="utf-8") as src:
            content = src.read()
        tmp_out = dest_path + ".tmp"
        with open(tmp_out, "w", encoding="utf-8") as out:
            out.write(content)
        os.replace(tmp_out, dest_path)
        # Remove original temp upload
        try:
            os.remove(tmp_path)
        except Exception as e:
            logger.info(f"Warning: failed to clean up temp file '{tmp_path}': {e}")
        logger.info(f"Stored updated user config for '{username}' at '{dest_path}'")
    except Exception as e:
        logger.info(f"Warning: failed to persist uploaded config to '{dest_path}': {e}")
        return False, username, None

    # Ensure conf contains this user file path
    conf_path = os.path.abspath(configure_file)
    ensure_user_file_in_conf(conf_path, username, dest_path)
    return True, username, dest_path
