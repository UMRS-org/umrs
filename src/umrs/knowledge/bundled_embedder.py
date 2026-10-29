"""Resolve a hub id to the image's bundled copy of that model (issue #635).

User configs name ``intfloat/e5-base-v2`` by hub id, so boot provisioning went
to the (empty, per-container) HF cache and then the network although the
umrs-server image carries the same revision at
``text_embedder.BUNDLED_TEXT_EMBEDDER_DIR``.

Kept out of ``text_embedder.py`` on purpose: ``docker/Dockerfile.server``
copies that file alone ahead of the source tree to stage the ~420 MB bundle
from Hugging Face in a cached layer. Any edit to it invalidates that layer and
re-downloads the bundle on the next image build.
"""

from __future__ import annotations

from umrs.knowledge import text_embedder


def bundled_dir_for(model_name: str) -> str | None:
    """The bundled copy's directory when ``model_name`` is the bundle's hub id.

    Any other name, or an image built without the bundle, returns None and the
    caller keeps the hub path. Reads the module attributes at call time so a
    test (or a site) can repoint ``BUNDLED_TEXT_EMBEDDER_DIR``.
    """
    bundle_dir = text_embedder.BUNDLED_TEXT_EMBEDDER_DIR
    if model_name != text_embedder.BUNDLED_TEXT_EMBEDDER_REPO:
        return None
    return bundle_dir if text_embedder.is_staged(bundle_dir) else None
