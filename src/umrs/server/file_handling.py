"""Server-side file operations: user config persistence, workflow upload."""

import json
import logging
import os
import re
import shutil
import struct
import zipfile

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
    """Ends an upload with this message."""


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


def _upload_dest(client_id: str, workflow_name: str) -> str:
    """The extraction directory, an ABSOLUTE path so that the stored
    ``source_path`` resolves regardless of the reader's working directory
    (the durable-job backend). UMRS_WORKFLOW_STATE_DIR relocates extracted
    workflows to a durable bind-mounted dir on deployed servers."""
    state_dir = os.environ.get("UMRS_WORKFLOW_STATE_DIR", "").strip()
    if state_dir:
        workflows_root: str = os.path.join(state_dir, "extracted", client_id)
    else:
        workflows_root = get_resolved_umrs_subdir(
            get_default_tmp_subdir(), "user_workflows", client_id
        )
    root = os.path.realpath(workflows_root)
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


def _package_check(raw: bytes) -> list[str]:
    """The upload contract (r4_contracts.workflow_package, 2026-10-02): a
    description and an explicit flow graph (every step's ``depends_on``)
    are mandatory, so an error refuses the upload, every problem named at
    once. Warnings come back as advice lines for the upload notice."""
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
    return [str(p) for p in warnings_of(problems)]


def _extract_checked(zf, dest_dir: str) -> list[str]:
    """Extract after every member is checked to stay inside *dest_dir*, the
    archive fits the source caps and a ``workflow.json`` (root or one level
    deep) meets the upload contract and parses. Returns the contract's
    warnings."""

    _check_zip_size(zf)
    for member in zf.namelist():
        if not _within(dest_dir, os.path.realpath(os.path.join(dest_dir, member))):
            raise _UploadRefused(f"Zip contains unsafe path: {member}")
    candidates = [n for n in zf.namelist() if os.path.basename(n) == "workflow.json"]
    if not candidates:
        raise _UploadRefused("Zip archive does not contain a workflow.json")
    for name in candidates:
        # Parsed whole into server memory: a small zip of a compressible
        # one expanded to max_src_bytes of JSON objects (~20 GiB of Python).
        declared = zf.getinfo(name).file_size
        if declared > MAX_WORKFLOW_JSON_BYTES:
            raise _UploadRefused(
                f"{name} is {declared} bytes (max {MAX_WORKFLOW_JSON_BYTES})"
            )
    warnings = _package_check(zf.read(candidates[0]))
    os.makedirs(dest_dir, exist_ok=True)
    zf.extractall(dest_dir)
    return warnings


def _flatten_nested(dest_dir: str) -> None:
    """A workflow.json one level deep (``myworkflow/workflow.json``): its
    directory's contents move up so *dest_dir* holds workflow.json."""
    if os.path.isfile(os.path.join(dest_dir, "workflow.json")):
        return
    for name in os.listdir(dest_dir):
        sub_dir = os.path.join(dest_dir, name)
        if os.path.isfile(os.path.join(sub_dir, "workflow.json")):
            for item in os.listdir(sub_dir):
                shutil.move(os.path.join(sub_dir, item), dest_dir)
            os.rmdir(sub_dir)
            return


def _mark_uploaded(dest_dir: str) -> None:
    """Rewrite workflow.json's source to the UPLOADED tree, which must lie
    inside *dest_dir*."""
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
    wf_data["source"] = {"type": "uploaded", "path": resolved_source}
    with open(actual_wf, "w", encoding="utf-8") as fh:
        json.dump(wf_data, fh, indent=2)


def handle_upload_workflow(client_id, zip_path, workflow_name):
    """Validate and extract an uploaded workflow zip to the user_workflows directory.

    The zip must contain a ``workflow.json`` at its root (or one level deep)
    that meets the upload contract (a description and every step's
    ``depends_on``, ``r4_contracts.workflow_package``) and parses as a
    :class:`WorkflowDefinition`; *workflow_name* must match
    :data:`WORKFLOW_NAME_RE` (one directory under the owner's tree).

    Returns ``(success: bool, dest_dir: str | None, message: str | None)``:
    the error on failure; on success an optional notice for the uploader
    (a Dockerfile in the upload is ignored, never built — ADR-074 D-7; the
    contract's advice, e.g. parameters no parameter_schema declares).
    """
    try:
        _precheck_upload(zip_path, workflow_name)
        dest_dir = _upload_dest(client_id, workflow_name)
    except _UploadRefused as refused:
        return False, None, str(refused)

    from umrs.server.step_staging import DEFAULT_CAPS

    try:
        with open(zip_path, "rb") as raw:
            guarded = _CountedDirectory(raw, 2 * DEFAULT_CAPS.max_src_files)
            with zipfile.ZipFile(guarded) as zf:
                guarded.armed = False
                advice = _extract_checked(zf, dest_dir)
        _flatten_nested(dest_dir)
        _mark_uploaded(dest_dir)
    except _UploadRefused as refused:
        return False, None, str(refused)
    except zipfile.BadZipFile:
        return False, None, f"Corrupted zip archive: {zip_path}"
    except Exception as e:
        return False, None, f"Failed to extract workflow: {e}"
    finally:
        # Clean up the uploaded temp zip
        try:
            os.remove(zip_path)
        except Exception as e:
            logger.info(f"Warning: failed to clean up zip file '{zip_path}': {e}")

    logger.info(
        f"Workflow '{workflow_name}' for user '{client_id}' extracted to {dest_dir}"
    )

    notices = [_dockerfile_notice(dest_dir, workflow_name), *advice]
    return True, dest_dir, "; ".join(n for n in notices if n) or None


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
