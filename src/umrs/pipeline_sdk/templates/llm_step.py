"""``LLMStep`` — built-in step that runs an LLM-reasoning prompt.

The step renders a Jinja-style prompt template against the outputs of
upstream steps (in particular ``VisionStep``) and calls the platform
LLM endpoint.  When ``output_schema`` is provided the step asks the
model to return JSON conforming to that schema and validates the
response.

The template language is intentionally minimal — only ``{{ steps.NAME.FIELD }}``
substitutions are supported, where ``FIELD`` is a dotted path into the
upstream step's ``StepResult.extra`` dict (e.g.
``{{ steps.vision_dense.dense_description }}``).  This avoids pulling
Jinja2 as a hard dependency while still supporting the common case.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


class LLMStep(Step):
    """Run a prompt template against a platform LLM endpoint."""

    name = "llm"

    def setup(self, config: dict[str, Any]) -> None:
        self.model: str = str(config.get("model", "rivault-llm-static"))
        self.system_prompt: str = str(config.get("system_prompt", "")).strip()
        self.prompt_template: str = str(config.get("prompt_template", ""))
        if not self.prompt_template:
            raise ValueError("LLMStep requires a non-empty prompt_template")
        self.output_schema: dict[str, Any] | None = config.get("output_schema")
        self.temperature: float = float(config.get("temperature", 0.0))
        self.max_tokens: int = int(config.get("max_tokens", 1024))
        self.timeout_sec: float = float(config.get("timeout_sec", 60.0))
        self.api_base: str | None = (
            config.get("api_base")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("R4_GATEWAY_URL")
        )
        self.api_key: str | None = config.get("api_key") or os.environ.get(
            "OPENAI_API_KEY"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, ctx: StepContext) -> StepResult:
        prompt = self._render_template(ctx)
        ctx.report_progress(10.0, "rendered LLM prompt")

        if os.environ.get("UMRS_LLM_MOCK", "0") in ("1", "true", "TRUE"):
            response_text = self._mock_response(prompt)
        else:
            try:
                response_text = self._call_llm(prompt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLMStep: LLM call failed (%s), returning mock response", exc
                )
                response_text = self._mock_response(prompt, fallback_error=str(exc))

        structured: dict[str, Any] | None = None
        validation_error: str | None = None
        if self.output_schema:
            structured, validation_error = self._extract_structured(response_text)

        extra: dict[str, Any] = {
            "prompt": prompt,
            "response_text": response_text,
            "model": self.model,
        }
        if structured is not None:
            extra["structured"] = structured
        if validation_error:
            extra["validation_error"] = validation_error

        # Persist response to S3 for the API's results endpoint.
        out_key = (
            f"{ctx.run_id or 'run'}/{ctx.config.get('step_name', 'llm')}/response.json"
        )
        try:
            payload = json.dumps(
                {
                    "prompt": prompt,
                    "response_text": response_text,
                    "structured": structured,
                    "validation_error": validation_error,
                    "model": self.model,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            ctx.upload_bytes(out_key, payload, content_type="application/json")
            output_keys: list[str] = [out_key]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMStep: failed to upload response JSON: %s", exc)
            output_keys = []

        ctx.report_progress(100.0, "llm_step complete")
        return StepResult(output_keys=output_keys, extra=extra)

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_template(self, ctx: StepContext) -> str:
        scope: dict[str, Any] = {
            "steps": {
                name: getattr(result, "extra", {}) or {}
                for name, result in (ctx.parent_results or {}).items()
            },
            "params": ctx.config,
        }

        def _replace(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            try:
                value = _resolve_dotted(scope, expr)
            except Exception:  # noqa: BLE001
                return match.group(0)
            if value is None:
                return ""
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        return _PLACEHOLDER_RE.sub(_replace, self.prompt_template)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is required for LLMStep") from exc

        if not self.api_base:
            raise RuntimeError(
                "LLMStep requires api_base (OPENAI_BASE_URL / R4_GATEWAY_URL)"
            )
        url = f"{self.api_base.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.output_schema:
            # Hint to the LLM that we want JSON output.
            body["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {data!r}") from exc
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content)

    def _mock_response(self, prompt: str, *, fallback_error: str | None = None) -> str:
        if self.output_schema:
            mock_obj = {
                "_mock": True,
                "prompt_chars": len(prompt),
            }
            if fallback_error:
                mock_obj["_error"] = fallback_error
            return json.dumps(mock_obj)
        return f"[mock LLM response] prompt had {len(prompt)} chars"

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _extract_structured(
        self, text: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Try to parse the model's text as JSON conforming to ``output_schema``.

        Returns ``(structured, error)``.  ``error`` is a string when the
        text is not parseable or fails very-light schema validation.
        """
        text = text.strip()
        # Tolerate fenced code blocks.
        if text.startswith("```"):
            text = text.strip("`")
            text = text.lstrip("json").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"
        if not isinstance(data, dict):
            return None, "structured output must be a JSON object"
        # Light required-field check (full JSON Schema validation is out of scope).
        required = (self.output_schema or {}).get("required", [])
        missing = [k for k in required if k not in data]
        if missing:
            return data, f"missing required keys: {missing}"
        return data, None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_dotted(scope: dict[str, Any], expr: str) -> Any:
    """Resolve ``a.b.c`` against ``scope``.  Returns ``None`` if not found."""
    current: Any = scope
    for part in expr.split("."):
        current = (
            current.get(part)
            if isinstance(current, dict)
            else getattr(current, part, None)
        )
        if current is None:
            return None
    return current
