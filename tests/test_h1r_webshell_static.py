"""H-1 web-shell lane: the /umrs/web static shell works behind /gw/web/.

In strict mode the dashboard API needs the gateway-verified identity, which
only the nginx ``/gw/web/`` location (auth_request + identity forwarding)
adds. The shell must therefore call the API under the prefix it was served
from, and its own assets must be relative so they load under either prefix.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

SHELL = Path(__file__).resolve().parents[1] / "src" / "umrs" / "server" / "web_gui"


def test_app_js_picks_the_gateway_api_base_when_served_under_gw():
    source = (SHELL / "app.js").read_text()
    assert (
        'const api = location.pathname.startsWith("/gw/") ? "/gw/web/dashboard" : "/umrs/web/dashboard";'
        in source
    )
    assert 'const api = "/umrs/web/dashboard";' not in source


def test_index_html_asset_links_are_relative():
    html = (SHELL / "index.html").read_text()
    refs = re.findall(r'(?:href|src)="([^"]+)"', html)
    assert sorted(refs) == ["app.css", "app.js"]


def test_relative_assets_resolve_under_either_mount():
    for prefix in ("/umrs/web", "/gw/web"):
        app = FastAPI()
        app.mount(prefix, StaticFiles(directory=str(SHELL), html=True), name="web")
        client = TestClient(app)
        page = client.get(f"{prefix}/")
        assert page.status_code == 200
        assert client.get(f"{prefix}/app.css").status_code == 200
        assert client.get(f"{prefix}/app.js").status_code == 200
