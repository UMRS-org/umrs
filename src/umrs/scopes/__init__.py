"""Signed scope tokens of work levels (ADR-108 D-3).

``tokens`` signs, verifies and narrows; ``store`` keeps revocations;
``umrs.api.scope_tokens`` issues and revokes, and reads ``x-r4-scope`` on the
routes a level calls.
"""
