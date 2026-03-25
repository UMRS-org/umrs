#!/usr/bin/env python3
"""Quick script to get full 502 response details."""
import http.client

conn = http.client.HTTPConnection("localhost", 3001, timeout=5)
try:
    conn.request("GET", "/readyz")
    resp = conn.getresponse()
    print(f"Status: {resp.status} {resp.reason}")
    for k, v in resp.getheaders():
        print(f"  {k}: {v}")
    body = resp.read().decode("utf-8", errors="replace")
    print(f"Body:\n{body[:2000]}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
finally:
    conn.close()
