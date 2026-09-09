"""Base-URL join for the inference proxy (design §3.3.2).

Target = origin(base_url) + prefix(base_url) + path, with **prefix
de-duplication**: when ``path`` already starts with the stored prefix, do
not repeat it (vLLM `https://host/v1` + request `/v1/chat/...` → single
/v1 prefix).
"""

from __future__ import annotations

from urllib.parse import urlsplit


def join_provider_url(base_url: str, path: str) -> str:
    """Join one request path onto a stored provider base URL.

    Always produces an absolute URL with the base's scheme/host; the query
    of the path is preserved verbatim unless ``base_url`` itself carries a
    query (then the path query wins — join never concatenates two queries).
    """
    if not base_url:
        raise ValueError("base_url required")
    parts = urlsplit(base_url)
    prefix = parts.path.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    joined_path = (
        path if path.startswith(prefix + "/") or path == prefix else prefix + path
    )
    return f"{parts.scheme}://{parts.netloc}{joined_path}"
