# Modality and task definitions for UMRS.
# Modality = data type (what the data is).
# Task     = processing type (what to do with the data).
#
# Must be checked by client and server.

import sys

# ── Pure data-type modalities ──────────────────────────────────────────────
VALID_MODALITIES = [
    "text",
    "text_file",
    "image",
    "volume",
    "movie",
    "volume_movie",
    "audio",
    "notebook",
    "code",
]

# ── Processing tasks ───────────────────────────────────────────────────────
# Well-known tasks built into UMRS.  Users may define additional custom tasks
# (e.g. for ``user_script://`` or ``bentoml://`` models) without modifying
# this list — unknown task strings are accepted as-is.
WELL_KNOWN_TASKS = [
    "embedding",
    "segmentation",
    "reconstruction",
    "mae_reconstruction",
    "feature_extraction",
    "detection",
    "classification",
]

# Alias kept for backward compatibility.
VALID_TASKS = WELL_KNOWN_TASKS

# Default task when none is specified.
DEFAULT_TASK = "embedding"

# ── Legacy compound names → (modality, task) ──────────────────────────────
# Old configs used a single ``modality_name`` that mixed data type and
# processing intent.  This mapping translates them into the new two-axis
# scheme.
LEGACY_MAPPING: dict[str, tuple[str, str]] = {
    # pure data types (task defaults to embedding)
    "image": ("image", "embedding"),
    "text": ("text", "embedding"),
    "text file": ("text_file", "embedding"),
    "text_file": ("text_file", "embedding"),
    "tomography": ("volume", "embedding"),
    # compound names
    "segmentation": ("image", "segmentation"),
    "tomography_embedding": ("volume", "embedding"),
    "tomography_reconstruction": ("volume", "reconstruction"),
    "tomography_mae_reconstruction": ("volume", "mae_reconstruction"),
    "tomography_features_extraction": ("volume", "feature_extraction"),
    "detection": ("image", "detection"),
}

# Set of all accepted legacy names (for fast lookup).
_LEGACY_NAMES = frozenset(LEGACY_MAPPING)

# ── Upload-category mapping (data type → upload behaviour) ────────────────
# The client uses this to decide *how* to push data to S3.
_MODALITY_UPLOAD_CATEGORY: dict[str, str] = {
    "text": "text",
    "text_file": "text_file",
    "image": "image",
    "volume": "image",
    "movie": "image",
    "volume_movie": "image",
    "audio": "audio",
    "notebook": "text_file",
    "code": "text_file",
}

# ── File extension → modality mapping ─────────────────────────────────────
# Used by ``detect_modality()`` to auto-detect the modality of a file.
_EXTENSION_MODALITY: dict[str, str] = {
    # Text
    ".txt": "text_file",
    ".md": "text_file",
    ".markdown": "text_file",
    ".rst": "text_file",
    ".pdf": "text_file",
    ".html": "text_file",
    ".htm": "text_file",
    ".json": "text_file",
    ".yaml": "text_file",
    ".yml": "text_file",
    ".csv": "text_file",
    ".tsv": "text_file",
    # Image
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
    # Video
    ".mp4": "movie",
    ".avi": "movie",
    ".mov": "movie",
    ".mkv": "movie",
    ".webm": "movie",
    ".flv": "movie",
    # Audio
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".m4a": "audio",
    ".aac": "audio",
    # Notebook
    ".ipynb": "notebook",
    # Code
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".jsx": "code",
    ".tsx": "code",
    ".rs": "code",
    ".go": "code",
    ".java": "code",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".hpp": "code",
    ".sh": "code",
    ".bash": "code",
    ".r": "code",
    ".R": "code",
    ".m": "code",
    ".jl": "code",
    ".rb": "code",
    ".sql": "code",
    ".scala": "code",
    ".swift": "code",
    ".kt": "code",
    ".kts": "code",
    ".dart": "code",
    ".lua": "code",
    ".pl": "code",
    ".pm": "code",
    ".t": "code",
    ".ps1": "code",
    ".bat": "code",
    ".cmd": "code",
    ".com": "code",
    ".fish": "code",
    ".zsh": "code",
    ".php": "code",
    ".phtml": "code",
    ".php3": "code",
    ".php4": "code",
    ".php5": "code",
    ".php7": "code",
    ".phps": "code",
    ".pht": "code",
    ".thtml": "code",
    ".pyp": "code",
    ".pyc": "code",
    ".pyo": "code",
    ".pyd": "code",
    ".pyw": "code",
    ".pyx": "code",
    ".pxi": "code",
    ".pxd": "code",
    ".pyz": "code",
    ".pywz": "code",
    ".egg": "code",
    ".whl": "code",
    ".zip": "code",
    ".tar": "code",
    ".gz": "code",
    ".bz2": "code",
    ".7z": "code",
    ".rar": "code",
    ".iso": "code",
    ".dmg": "code",
    ".pkg": "code",
    ".deb": "code",
    ".rpm": "code",
    ".msi": "code",
    ".apk": "code",
    ".jar": "code",
    ".war": "code",
    ".ear": "code",
    ".class": "code",
    ".py": "code",
}


# ── Public helpers ─────────────────────────────────────────────────────────


def resolve_modality_task(
    modality_name: str,
    task: str | None = None,
) -> tuple[str, str]:
    """Resolve a (possibly legacy) modality_name + optional task into a
    canonical ``(modality, task)`` pair.

    Accepts three input forms:
    1. New-style:  ``modality_name`` is a pure data type, ``task`` is given.
    2. New-style with default:  ``modality_name`` is pure, ``task`` is ``None``
       → task defaults to ``"embedding"``.
    3. Legacy:  ``modality_name`` is a compound string present in
       ``LEGACY_MAPPING``.  ``task`` is ignored.

    Raises ``ValueError`` for unrecognised modality names.
    """
    # When an explicit task is provided and the name is a valid pure
    # modality, prefer the new-style path — even if the name also appears
    # in LEGACY_MAPPING (e.g. "image", "text").
    if modality_name in VALID_MODALITIES:
        resolved_task = task if task else DEFAULT_TASK
        return modality_name, resolved_task

    # Legacy compound names (e.g. "segmentation", "tomography_reconstruction")
    if modality_name in LEGACY_MAPPING:
        return LEGACY_MAPPING[modality_name]

    raise ValueError(
        f"Unknown modality '{modality_name}'. "
        f"Valid modalities: {VALID_MODALITIES}. "
        f"Legacy names: {sorted(_LEGACY_NAMES)}"
    )


def modality_category(modality_name: str) -> str:
    """Return the upload category for a given modality name.

    Accepts both legacy compound names and new pure data types.
    Falls back to ``"image"`` for unknown modalities so that new
    vision sub-categories work without client changes.
    """
    # Resolve to pure modality first, then look up upload category.
    try:
        pure_modality, _ = resolve_modality_task(modality_name)
    except ValueError:
        return "image"
    return _MODALITY_UPLOAD_CATEGORY.get(pure_modality, "image")


def is_valid_modality(modality_name: str) -> bool:
    """Return True if *modality_name* is a known pure modality or legacy name."""
    return modality_name in VALID_MODALITIES or modality_name in _LEGACY_NAMES


def check_or_quit(modality_name: str) -> None:
    if not is_valid_modality(modality_name):
        sys.exit(f"Trying to use unsupported modality {modality_name}\nExiting")


# ── Automatic modality detection ──────────────────────────────────────────


def detect_modality(file_path: str) -> str:
    """Auto-detect the modality of a file from its extension.

    Falls back to ``"text_file"`` for unknown extensions so that
    new file types are treated as text by default.

    Examples:
        >>> detect_modality("paper.pdf")
        'text_file'
        >>> detect_modality("image.png")
        'image'
        >>> detect_modality("recording.wav")
        'audio'
        >>> detect_modality("analysis.ipynb")
        'notebook'
        >>> detect_modality("script.py")
        'code'
    """
    import os

    ext = os.path.splitext(file_path)[1].lower()
    return _EXTENSION_MODALITY.get(ext, "text_file")


def is_supported_file(file_path: str) -> bool:
    """Return True if the file extension is recognized by ``detect_modality``."""
    import os

    ext = os.path.splitext(file_path)[1].lower()
    return ext in _EXTENSION_MODALITY
