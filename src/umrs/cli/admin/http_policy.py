"""``umrs-server admin policy …``: session policy limit rules (ADR-099 D-3).

Calls the running UMRS (``--umrs-url`` / ``$UMRS_BASE_URL``) with the admin
JWT of ``admin session`` (``http_admin.admin_jwt``):

    umrs-server admin policy list      [--venue V | --all-sites] [--scope S]
    umrs-server admin policy show      --scope S [--subject X] [--venue V | --all-sites]
    umrs-server admin policy set       --scope S [--subject X] [--venue V | --all-sites]
                                       [--note T] [--replace] (--file doc.json | KEY=VALUE ...)
    umrs-server admin policy unset     --scope S [--subject X] [--venue V | --all-sites]
    umrs-server admin policy effective --user U [--project P] [--venue V]

``KEY=VALUE``: ``section.field=value``, e.g. ``budget.tokens_per_day_user=2000000``
or ``web.allow=none,allowlist``; an empty value or ``null`` sets null (no
limit at this layer). ``set`` merges the pairs into the stored rule unless
``--replace``; ``--file`` replaces the whole document. ``--venue`` defaults
to the server's own site; ``--all-sites`` writes the rule every site reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, NoReturn
from urllib.parse import quote, urlencode

from umrs.session_policy.model import RULE_FIELDS, SCOPES

from .http_admin import _print_table, umrs_api, umrs_base_url

_RULE_COLS = ("VENUE", "SCOPE", "SUBJECT", "LAYER", "SET BY", "UPDATED", "DOC")


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


# ── KEY=VALUE documents ────────────────────────────────────────────────────


def _field_kind(key: str) -> str:
    section, _, name = key.partition(".")
    kind = RULE_FIELDS.get(section, {}).get(name)
    if kind is None:
        known = ", ".join(
            f"{s}.{f}" for s, fields in RULE_FIELDS.items() for f in fields
        )
        _fail(f"unknown policy field {key!r}; known: {known}")
    return kind


def parse_value(key: str, raw: str) -> Any:
    """One ``KEY=VALUE`` value: null, a comma list (modes, domains) or an int."""
    kind = _field_kind(key)
    text = raw.strip()
    if text in ("", "null"):
        return None
    if kind in ("modes", "domains"):
        return [part.strip() for part in text.split(",") if part.strip()]
    try:
        return int(text)
    except ValueError:
        _fail(f"{key} takes an integer, an empty value or null (got {raw!r})")


def parse_assignments(pairs: list[str]) -> dict[str, dict[str, Any]]:
    """``["budget.rpm_user=60", "web.allow=none,allowlist"]`` → a rule doc."""
    doc: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            _fail(f"expected KEY=VALUE, got {pair!r}")
        key = key.strip()
        section, _, name = key.partition(".")
        doc.setdefault(section, {})[name] = parse_value(key, raw)
    return doc


def merge_docs(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """``update``'s fields over ``base`` (section by section)."""
    merged = {section: dict(fields) for section, fields in base.items()}
    for section, fields in update.items():
        merged.setdefault(section, {}).update(fields)
    return merged


# ── HTTP ───────────────────────────────────────────────────────────────────


def _base(args: argparse.Namespace) -> str:
    return f"{umrs_base_url(args)}/umrs/admin/session-policy"


def _venue(args: argparse.Namespace) -> str | None:
    """None = the server's own site; "" = every site."""
    if getattr(args, "all_sites", False):
        return ""
    return args.venue


def _subject(args: argparse.Namespace) -> str:
    if args.scope == "site":
        return "_"
    if not (args.subject or "").strip():
        _fail(f"--subject names the {args.scope} of a {args.scope} rule")
    return args.subject.strip()


def _query(**params: Any) -> str:
    kept = {k: v for k, v in params.items() if v is not None}
    return f"?{urlencode(kept)}" if kept else ""


def _rules(args: argparse.Namespace, *, scope: str | None) -> dict[str, Any]:
    url = _base(args) + _query(venue=_venue(args), scope=scope)
    return umrs_api("GET", url, args) or {}


def _matching(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The stored rules at ``--scope``/``--subject`` and the chosen venue."""
    data = _rules(args, scope=args.scope)
    subject = "" if args.scope == "site" else _subject(args)
    venue = _venue(args)
    wanted = data.get("site", "") if venue is None else venue
    return [
        r
        for r in data.get("rules", [])
        if r.get("subject") == subject and r.get("venue") == wanted
    ]


def _rule_row(rule: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(rule.get("venue") or "(all)"),
        str(rule.get("scope", "")),
        str(rule.get("subject") or "-"),
        str(rule.get("layer", "")),
        str(rule.get("set_by") or "-"),
        str(rule.get("updated_at") or "-")[:19],
        json.dumps(rule.get("doc") or {}, separators=(",", ":"))[:80],
    )


# ── commands ───────────────────────────────────────────────────────────────


def _cmd_policy_list(args: argparse.Namespace, _store: Any) -> int:
    """Every stored rule (both layers)."""
    rules = _rules(args, scope=args.scope or None).get("rules", [])
    if args.json:
        print(json.dumps(rules, indent=2, default=str))
        return 0
    if not rules:
        print("(no session policy rules: every session gets the site defaults)")
        return 0
    _print_table(_RULE_COLS, [_rule_row(r) for r in rules])
    return 0


def _cmd_policy_show(args: argparse.Namespace, _store: Any) -> int:
    """The rules at one key (admin and owner layers)."""
    print(json.dumps(_matching(args), indent=2, default=str))
    return 0


def _new_doc(args: argparse.Namespace) -> dict[str, Any]:
    if args.file:
        if args.pairs:
            _fail("give --file or KEY=VALUE pairs, not both")
        with open(args.file, encoding="utf-8") as handle:
            return json.load(handle)
    if not args.pairs:
        _fail("nothing to set: give KEY=VALUE pairs or --file doc.json")
    update = parse_assignments(args.pairs)
    if args.replace:
        return update
    stored = next((r for r in _matching(args) if r.get("layer") == "admin"), None)
    return merge_docs((stored or {}).get("doc") or {}, update)


def _cmd_policy_set(args: argparse.Namespace, _store: Any) -> int:
    """Write the admin rule at one key (merging KEY=VALUE pairs by default)."""
    body: dict[str, Any] = {"doc": _new_doc(args), "note": args.note}
    venue = _venue(args)
    if venue is not None:
        body["venue"] = venue
    url = f"{_base(args)}/{args.scope}/{quote(_subject(args), safe='')}"
    print(json.dumps(umrs_api("PUT", url, args, body), indent=2, default=str))
    return 0


def _cmd_policy_unset(args: argparse.Namespace, _store: Any) -> int:
    """Remove the admin rule at one key."""
    url = f"{_base(args)}/{args.scope}/{quote(_subject(args), safe='')}"
    umrs_api("DELETE", url + _query(venue=_venue(args), layer=args.layer), args)
    print(f"removed the {args.layer} rule {args.scope}/{_subject(args)}")
    return 0


def _cmd_policy_effective(args: argparse.Namespace, _store: Any) -> int:
    """What a start of --user in --project would get (before the web choice)."""
    url = f"{_base(args)}/effective" + _query(
        user=args.user, project=args.project, venue=args.venue
    )
    print(json.dumps(umrs_api("GET", url, args), indent=2, default=str))
    return 0


# ── parsers ────────────────────────────────────────────────────────────────


def _add_where(parser: argparse.ArgumentParser, *, scope_required: bool) -> None:
    parser.add_argument(
        "--scope", choices=list(SCOPES), required=scope_required, default=None
    )
    parser.add_argument("--subject", default="", help="the project or user id")
    sites = parser.add_mutually_exclusive_group()
    sites.add_argument(
        "--venue", default=None, help="site name (default: the server's)"
    )
    sites.add_argument(
        "--all-sites", action="store_true", help="the rule every site reads (venue '')"
    )


def add_policy_parsers(policy_sub: argparse._SubParsersAction) -> None:
    list_p = policy_sub.add_parser("list", help="List the stored limit rules")
    _add_where(list_p, scope_required=False)
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=_cmd_policy_list)

    show_p = policy_sub.add_parser("show", help="Show the rules at one key")
    _add_where(show_p, scope_required=True)
    show_p.set_defaults(func=_cmd_policy_show)

    set_p = policy_sub.add_parser(
        "set",
        help="Write an admin rule (KEY=VALUE pairs or --file)",
        description=(
            "KEY=VALUE: section.field=value, e.g. budget.tokens_per_day_user=2000000 "
            "or web.allow=none,allowlist; an empty value or null = no limit at this "
            "layer. Pairs merge into the stored rule unless --replace."
        ),
    )
    _add_where(set_p, scope_required=True)
    set_p.add_argument("pairs", nargs="*", metavar="KEY=VALUE")
    set_p.add_argument("--file", default=None, help="a JSON rule document (replaces)")
    set_p.add_argument("--replace", action="store_true", help="replace, do not merge")
    set_p.add_argument("--note", default="", help="why (kept with the rule)")
    set_p.set_defaults(func=_cmd_policy_set)

    unset_p = policy_sub.add_parser("unset", help="Remove the rule at one key")
    _add_where(unset_p, scope_required=True)
    unset_p.add_argument("--layer", choices=["admin", "owner"], default="admin")
    unset_p.set_defaults(func=_cmd_policy_unset)

    eff_p = policy_sub.add_parser(
        "effective", help="Show the limits a session of a user would get"
    )
    eff_p.add_argument("--user", required=True)
    eff_p.add_argument("--project", default="")
    eff_p.add_argument(
        "--venue", default=None, help="site name (default: the server's)"
    )
    eff_p.set_defaults(func=_cmd_policy_effective)
