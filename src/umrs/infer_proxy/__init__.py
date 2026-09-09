"""Credential-injecting inference proxy (R1-2, D-1/D-3/D-6).

Sits in the UMRS trust zone. Holds the ONLY unseal private key. Per
request: re-validates the session-token lease from Postgres (no cache),
enforces the provider path policy, unseals the provider credential, injects
it, and streams. The worker never sees provider keys.
"""
