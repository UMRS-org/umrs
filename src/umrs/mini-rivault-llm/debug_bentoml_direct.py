#!/usr/bin/env python3
"""Connect directly to BentoML container IP."""
import http.client

conn = http.client.HTTPConnection("192.168.4.35", 3000, timeout=3)
try:
    conn.request("GET", "/readyz")
    resp = conn.getresponse()
    print("Status:", resp.status, resp.reason)
except Exception as e:
    print("Error:", type(e).__name__, e)
finally:
    conn.close()
