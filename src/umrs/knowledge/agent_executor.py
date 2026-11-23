"""UMRS-side agent iteration executor (Wave 9, parity with r4 LocalAgentIterationExecutor).

Each iteration of a UMRS research loop becomes one bounded agent turn against
the site's litellm upstream (``LITELLM_MASTER_KEY`` / service key, same
auth path as api/llm.py). The decision contract is identical to the r4 local
executor: one hypothesis grounded in the project knowledge graph, evidence
ids, an evaluation, a next step, and an optional W8 ``requires`` hint — as a
JSON object.

When the chat call fails or is unparseable, the iteration degrades to a
digest entry (``llm: false``) and the loop stays durable rather than wedging.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from . import llm_chat

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

AGENT_SYSTEM_PROMPT = (
    "You are a research-loop iteration agent for an autonomous scientific "
    "research runtime. On each iteration you are given: the research goal, a "
    "digest of the current project knowledge graph, and the hypotheses proposed "
    "so far. Propose exactly ONE new hypothesis grounded in the graph, answer "
    "with a JSON object only — no prose outside the JSON. Schema:\n"
    '{"hypothesis": "one-sentence hypothesis",'
    ' "evidence_ids": ["ids or labels of supporting knowledge nodes"],'
    ' "evaluation": "1-3 sentence progress evaluation vs the goal",'
    ' "next_step": "the next concrete action (ingest/experiment/query)",'
    ' "requires": {} or {"gpu": 1, "cores": N, "cluster": true} — optional, '
    "only when THIS step would actually need cluster-class compute}"
)

_CLUSTER_REQUIRE_KEYS = {"gpu", "cores", "cluster", "slurm", "mpi", "fugaku"}


def normalize_iteration_requires(raw: Any) -> dict[str, Any]:
    """Cluster-flavored subset of per-iteration requires (W8 contract)."""
    if not isinstance(raw, dict):
        return {}
    return {
        str(k).lower(): v
        for k, v in raw.items()
        if str(k).lower() in _CLUSTER_REQUIRE_KEYS
        and v not in (None, False, "", 0, "0", "false")
    }


def _node_digest(node: Any) -> dict[str, str]:
    """Compact node row for the LLM prompt digest (#207)."""
    node_type = getattr(node, "node_type", None)
    return {
        "id": str(getattr(node, "node_id", getattr(node, "id", "?"))),
        "type": str(getattr(node_type, "value", node_type) or "knowledge"),
        "name": str(getattr(node, "title", getattr(node, "name", "?"))),
    }


def _edge_digest(edge: Any) -> dict[str, str] | None:
    """Compact edge row for the LLM prompt digest (#207)."""
    source = getattr(edge, "source_id", None)
    target = getattr(edge, "target_id", None)
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    rel = getattr(edge, "relation_type", None)
    return {
        "from": source,
        "to": target,
        "type": str(getattr(rel, "value", rel) or "related"),
    }


class UmrsAgentIterationExecutor:
    """Per-iteration agent for UMRS loops (callable as executor).

    Signature mirrors the r4 LocalRuntime executor: ``(loop_id, iteration,
    research_goal) -> summary dict``.
    """

    def __init__(
        self,
        project: Any,
        *,
        chat: Callable[..., str] | None = None,
        model: str | None = None,
        max_tokens: int = 700,
        max_nodes: int = 30,
        project_graph_loader: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._project = project
        self._chat = chat
        # The model a caller or the site named; empty = whatever the registry serves.
        self._wanted_model = (model or os.getenv("UMRS_AGENT_MODEL", "")).strip()
        self._model = self._wanted_model or "rivault-gpt-oss-static"
        self._max_tokens = max_tokens
        self._max_nodes = max_nodes
        self._graph_loader = project_graph_loader
        self.hypotheses: list[dict[str, Any]] = []

    # project may be a ResearchProject or a plain object with knowledge_graph attr
    def _load_graph(self) -> dict[str, Any]:
        loader = self._graph_loader
        if loader is not None:
            try:
                return loader()
            except Exception as exc:  # noqa: BLE001
                return {"nodes": [], "edges": [], "note": f"graph loader failed: {exc}"}
        graph = getattr(self._project, "knowledge_graph", None)
        if graph is None:
            return {"nodes": [], "edges": [], "note": "knowledge graph unavailable"}
        # #207: read public KnowledgeGraph.nodes/.edges — the previous private
        # `_nodes` attr never existed, and edges were hardcoded [], so every
        # loop silently ran with an empty graph digest.
        try:
            nodes = list(getattr(graph, "nodes", {}).values())
            edge_rows = [
                row
                for row in (_edge_digest(e) for e in getattr(graph, "edges", []))
                if row is not None
            ]
        except Exception:  # noqa: BLE001
            return {"nodes": [], "edges": [], "note": "graph traversal failed"}
        return {"nodes": [_node_digest(n) for n in nodes], "edges": edge_rows}

    def __call__(
        self, loop_id: str, iteration_number: int, research_goal: str
    ) -> dict[str, Any]:
        del loop_id  # identity tracked via hypotheses journal
        graph = self._load_graph()
        reply = self._call_llm(research_goal, graph, iteration_number)
        return self._build_summary(iteration_number, research_goal, graph, reply)

    # ── llm ────────────────────────────────────────────────────────────

    def _messages(
        self, goal: str, graph: dict[str, Any], number: int
    ) -> list[dict[str, Any]]:
        context = self._graph_context(graph)
        if self.hypotheses:
            context += "\nHypotheses already proposed (do NOT repeat these):\n"
            context += "\n".join(f"- {h['hypothesis']}" for h in self.hypotheses[-5:])
        return [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research goal: {goal}\nIteration: {number}\n"
                    f"Knowledge graph digest:\n{context}"
                ),
            },
        ]

    def _graph_context(self, graph: dict[str, Any]) -> str:
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        lines = [f"nodes={len(nodes)} edges={len(edges)}"]
        for node in nodes[: self._max_nodes]:
            label = node.get("name") or node.get("title") or node.get("id") or "?"
            node_type = node.get("type", "knowledge")
            lines.append(f"- [{node_type}] {label}")
        return "\n".join(lines)

    def _call_llm(
        self, goal: str, graph: dict[str, Any], number: int
    ) -> dict[str, Any] | None:
        chat, model = self._chat, self._model
        if chat is None:
            resolved = self._registry_chat()
            if isinstance(resolved, dict):
                return resolved
            chat, model = resolved
        try:
            content = chat(
                self._messages(goal, graph, number),
                model=model,
                max_tokens=self._max_tokens,
            )
            return self._parse_reply(str(content or ""))
        except Exception as exc:  # noqa: BLE001 - digest-only iteration, keep loop durable
            return {"_error": str(exc)}

    def _registry_chat(self) -> tuple[Callable[..., str], str] | dict[str, Any]:
        """(chat callable, model) for this loop's turns, or the ``_error`` reply."""
        from umrs.orchestrator.registry_route import RegistryRouteRefused

        try:
            target = self._litellm_target()
        except RegistryRouteRefused as exc:
            return {"_error": str(exc)}
        if target is None:
            return {
                "_error": "no READY chat endpoint in the registry "
                "and UMRS_AGENT_LITELLM_URL unset"
            }
        base_url, model = target
        key = llm_chat.resolve_service_key()
        chat = lambda messages, model, max_tokens: llm_chat.chat_completion(
            messages,
            model=model,
            base_url=base_url,
            api_key=key,
            max_tokens=max_tokens,
        )
        return chat, model

    def _litellm_target(self) -> tuple[str, str] | None:
        """(base_url, model) for this loop's turns, or None when none is usable.

        An explicit UMRS_AGENT_LITELLM_URL wins (the resolver's one declared
        exception, DEC-063), so sites that configured it keep their
        behaviour. Otherwise the project's chat endpoint comes from the gated
        registry resolver (ADR-068, gap D-63), and a named model picks among
        the candidates; a sensitive project whose only rows are public
        passthroughs raises ``RegistryRouteRefused`` (2026-11-24). The old
        default, http://127.0.0.1:11434, is the container itself inside one:
        every agentic iteration failed there and was still marked completed.
        """
        from umrs.jobs.knowledge_adapters import _registry_chat_candidates
        from umrs.orchestrator import registry_route

        project_id = str(getattr(self._project, "project_id", "") or "").strip()
        explicit = registry_route.agent_url_exception(project_id)
        if explicit:
            return explicit, self._model
        if not project_id:
            # An empty project would make the registry lookup unfiltered:
            # every project's private endpoint would count as "own".
            return None
        endpoint = registry_route.resolve_registry_endpoint(
            project_id,
            self._wanted_model or None,
            rows=_registry_chat_candidates(project_id),
        )
        if endpoint is None:
            return None
        return endpoint.litellm_url, registry_route.served_name(endpoint)

    @staticmethod
    def _parse_reply(text: str) -> dict[str, Any]:
        match = _JSON_BLOCK_RE.search(text)
        if match is None:
            return {"_parse_error": True, "raw": text[:200]}
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return {"_parse_error": True, "raw": text[:200]}
        return (
            payload
            if isinstance(payload, dict)
            else {"_parse_error": True, "raw": text[:200]}
        )

    # ── summary ───────────────────────────────────────────────────────

    def _build_summary(
        self,
        number: int,
        goal: str,
        graph: dict[str, Any],
        reply: dict[str, Any] | None,
    ) -> dict[str, Any]:
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        summary: dict[str, Any] = {
            "iteration": number,
            "phase": "agent_iteration",
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
        }
        if reply and "hypothesis" in reply:
            return self._apply_decision(summary, number, reply)
        summary["llm"] = False
        summary["note"] = (reply or {}).get(
            "_error"
        ) or "LLM reply unusable; digest-only iteration"
        return summary

    def _apply_decision(
        self, summary: dict[str, Any], number: int, reply: dict[str, Any]
    ) -> dict[str, Any]:
        """Record the LLM decision into the hypotheses journal + summary."""
        hypothesis = str(reply.get("hypothesis", "")).strip()
        requires = normalize_iteration_requires(reply.get("requires"))
        if hypothesis:
            entry: dict[str, Any] = {
                "iteration": number,
                "hypothesis": hypothesis,
                "evidence_ids": reply.get("evidence_ids", []),
            }
            if requires:
                entry["requires"] = requires
            self.hypotheses.append(entry)
        summary.update(
            {
                "llm": True,
                "hypothesis": hypothesis,
                "evidence_ids": reply.get("evidence_ids", []),
                "evaluation": reply.get("evaluation", ""),
                "next_step": reply.get("next_step", ""),
            }
        )
        if requires:
            summary["requires"] = requires
        return summary
