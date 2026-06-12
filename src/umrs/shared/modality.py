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
