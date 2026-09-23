"""Retired Singularity wrapper — the refusal shell (ADR-061).

2026-09-23: the unreachable body (launch/stop/inspect and the attribute
assignments after the constructor raise) is deleted; the class remains
importable so references fail at CONSTRUCTION with the named error,
never an ImportError smudge. The live Apptainer lane (AI4S site,
runtime.kind: singularity) rides site_profiles/rivault_launcher — not
this wrapper.
"""


class SingularityWrapper:
    """Retired (ADR-061): the mini-RiVault sandbox is Podman-only.

    Constructing raises by name; there is no working configuration.
    """

    def __init__(
        self,
        default_image: str | None = None,
        bind_paths: list | None = None,
        nvidia_support: bool = False,
    ):
        # ADR-061 (2026-09-22): the Singularity sandbox path is retired.
        # The class remains importable so references fail at CONSTRUCTION
        # with a named error instead of an ImportError smudge.
        raise RuntimeError(
            "SingularityWrapper is retired (ADR-061): the mini-RiVault sandbox "
            "is Podman-only — use the podman-compose wrapper / slurm-remote lane."
        )

    @property
    def runtime_name(self) -> str:
        """The refusal surface: profile renders still name the runtime."""
        return "singularity"
