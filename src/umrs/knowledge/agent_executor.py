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


def _resolve_litellm_service_key() -> str | None:
    return os.environ.get("GATEWAY_LITELLM_SERVICE_KEY") or os.environ.get(
        "LITELLM_MASTER_KEY"
    )


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


def _default_litellm_chat(
    messages: list[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    max_tokens: int = 700,
    timeout: float = 60.0,
) -> str:
    """Chat via the site LiteLLM upstream (same auth as api/llm.py)."""
    key = api_key or _resolve_litellm_service_key()
    if not base_url:
        raise RuntimeError("no litellm base url configured for agent executor")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    import httpx

    resp = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=timeout,
        trust_env=False,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data["choices"][0]["message"]["content"])


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
        self._model = model or os.getenv("UMRS_AGENT_MODEL", "rivault-gpt-oss-static")
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
        chat = self._chat
        if chat is None:
            base_url = os.getenv("UMRS_AGENT_LITELLM_URL", "http://127.0.0.1:11434")
            key = _resolve_litellm_service_key()
            chat = lambda messages, model, max_tokens: _default_litellm_chat(
                messages,
                model=model,
                base_url=base_url,
                api_key=key,
                max_tokens=max_tokens,
            )
        try:
            content = chat(
                self._messages(goal, graph, number),
                model=self._model,
                max_tokens=self._max_tokens,
            )
            return self._parse_reply(str(content or ""))
        except Exception as exc:  # noqa: BLE001 - digest-only iteration, keep loop durable
            return {"_error": str(exc)}

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
