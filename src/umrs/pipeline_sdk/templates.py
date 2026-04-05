"""
B7 — Pipeline Templates

Catalog of reusable workflow templates for common analysis patterns.
Templates are parameterized workflow.json skeletons that users fill in.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ParameterType(str, Enum):
    """Types for template parameters."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    PATH = "path"


@dataclass
class TemplateParameter:
    """Typed parameter with defaults and validation."""
    name: str
    param_type: ParameterType
    description: str = ""
    default: Any = None
    required: bool = True
    choices: List[str] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def validate(self, value: Any) -> Optional[str]:
        """Return error message if invalid, None if valid."""
        if value is None:
            return f"Parameter '{self.name}' is required" if self.required else None

        if self.param_type == ParameterType.INTEGER:
            if not isinstance(value, int):
                return f"'{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return f"'{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return f"'{self.name}' must be <= {self.max_value}"

        elif self.param_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                return f"'{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return f"'{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return f"'{self.name}' must be <= {self.max_value}"

        elif self.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return f"'{self.name}' must be a boolean"

        elif self.param_type == ParameterType.CHOICE:
            if self.choices and value not in self.choices:
                return f"'{self.name}' must be one of {self.choices}"

        return None


@dataclass
class PipelineTemplate:
    """Parameterized template definition."""
    template_id: str
    name: str
    description: str
    parameters: List[TemplateParameter] = field(default_factory=list)
    workflow_skeleton: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = ""

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate all parameters. Returns list of errors."""
        errors = []
        for p in self.parameters:
            val = params.get(p.name, p.default)
            err = p.validate(val)
            if err:
                errors.append(err)
        return errors

    def render(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Render workflow.json by substituting parameters into skeleton."""
        errors = self.validate_params(params)
        if errors:
            raise ValueError(f"Invalid parameters: {'; '.join(errors)}")

        # Merge defaults
        full_params = {}
        for p in self.parameters:
            full_params[p.name] = params.get(p.name, p.default)

        # Deep-copy skeleton and substitute
        workflow = copy.deepcopy(self.workflow_skeleton)
        return self._substitute(workflow, full_params)

    def _substitute(self, obj: Any, params: Dict[str, Any]) -> Any:
        if isinstance(obj, str):
            # Replace {{param_name}} placeholders
            for key, value in params.items():
                placeholder = "{{" + key + "}}"
                if placeholder in obj:
                    if obj == placeholder:
                        return value
                    obj = obj.replace(placeholder, str(value))
            return obj
        elif isinstance(obj, dict):
            return {k: self._substitute(v, params) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute(item, params) for item in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "parameters": [asdict(p) for p in self.parameters],
            "workflow_skeleton": self.workflow_skeleton,
            "tags": self.tags,
            "version": self.version,
            "author": self.author,
        }


# ---------- Built-in Templates ----------

def _image_embedding_template() -> PipelineTemplate:
    return PipelineTemplate(
        template_id="image_embedding",
        name="Image Embedding Pipeline",
        description="Upload images → compute embeddings → export CSV",
        tags=["image", "embedding", "vision"],
        parameters=[
            TemplateParameter("input_bucket", ParameterType.STRING, "S3 bucket with input images"),
            TemplateParameter("output_bucket", ParameterType.STRING, "S3 bucket for results"),
            TemplateParameter("model_name", ParameterType.STRING, "Embedding model name", default="hmae-pretrained"),
            TemplateParameter("batch_size", ParameterType.INTEGER, "Images per batch", default=16, min_value=1, max_value=256),
        ],
        workflow_skeleton={
            "name": "image-embedding",
            "source": {"type": "UPLOADED", "path": "."},
            "execution_mode": "IN_PROCESS",
            "steps": [
                {"name": "load_images", "entrypoint": "steps/load.py:LoadImages", "config": {"bucket": "{{input_bucket}}"}},
                {"name": "compute_embeddings", "entrypoint": "steps/embed.py:Embed", "depends_on": ["load_images"],
                 "config": {"model": "{{model_name}}", "batch_size": "{{batch_size}}"}},
                {"name": "export_csv", "entrypoint": "steps/export.py:ExportCSV", "depends_on": ["compute_embeddings"],
                 "config": {"output_bucket": "{{output_bucket}}"}},
            ],
        },
    )


def _text_analysis_template() -> PipelineTemplate:
    return PipelineTemplate(
        template_id="text_analysis",
        name="Text Analysis Pipeline",
        description="Upload text → tokenize → embed → semantic search",
        tags=["text", "embedding", "nlp", "rag"],
        parameters=[
            TemplateParameter("input_bucket", ParameterType.STRING, "S3 bucket with text files"),
            TemplateParameter("output_bucket", ParameterType.STRING, "S3 bucket for results"),
            TemplateParameter("model_name", ParameterType.STRING, "Embedding model", default="intfloat/e5-base-v2"),
            TemplateParameter("chunk_size", ParameterType.INTEGER, "Characters per chunk", default=512, min_value=64),
        ],
        workflow_skeleton={
            "name": "text-analysis",
            "source": {"type": "UPLOADED", "path": "."},
            "execution_mode": "IN_PROCESS",
            "steps": [
                {"name": "load_text", "entrypoint": "steps/load_text.py:LoadText", "config": {"bucket": "{{input_bucket}}"}},
                {"name": "chunk", "entrypoint": "steps/chunk.py:ChunkText", "depends_on": ["load_text"],
                 "config": {"chunk_size": "{{chunk_size}}"}},
                {"name": "embed", "entrypoint": "steps/embed_text.py:EmbedText", "depends_on": ["chunk"],
                 "config": {"model": "{{model_name}}"}},
                {"name": "index", "entrypoint": "steps/index.py:BuildIndex", "depends_on": ["embed"],
                 "config": {"output_bucket": "{{output_bucket}}"}},
            ],
        },
    )


def _defect_detection_template() -> PipelineTemplate:
    return PipelineTemplate(
        template_id="defect_detection",
        name="Defect Detection Pipeline",
        description="Upload images → segment → classify defects",
        tags=["image", "segmentation", "defect", "sam3"],
        parameters=[
            TemplateParameter("input_bucket", ParameterType.STRING, "S3 bucket with inspection images"),
            TemplateParameter("output_bucket", ParameterType.STRING, "S3 bucket for results"),
            TemplateParameter("threshold", ParameterType.FLOAT, "Detection threshold", default=0.5, min_value=0.0, max_value=1.0),
        ],
        workflow_skeleton={
            "name": "defect-detection",
            "source": {"type": "UPLOADED", "path": "."},
            "execution_mode": "IN_PROCESS",
            "steps": [
                {"name": "load", "entrypoint": "steps/load.py:LoadImages", "config": {"bucket": "{{input_bucket}}"}},
                {"name": "segment", "entrypoint": "steps/segment.py:Segment", "depends_on": ["load"],
                 "config": {"model": "sam3"}},
                {"name": "classify", "entrypoint": "steps/classify.py:ClassifyDefects", "depends_on": ["segment"],
                 "config": {"threshold": "{{threshold}}", "output_bucket": "{{output_bucket}}"}},
            ],
        },
    )


def _video_analysis_template() -> PipelineTemplate:
    return PipelineTemplate(
        template_id="video_analysis",
        name="Video Analysis Pipeline",
        description="Upload video → extract frames → analyze",
        tags=["video", "frames", "analysis"],
        parameters=[
            TemplateParameter("input_bucket", ParameterType.STRING, "S3 bucket with videos"),
            TemplateParameter("output_bucket", ParameterType.STRING, "S3 bucket for results"),
            TemplateParameter("fps", ParameterType.INTEGER, "Frames per second to extract", default=1, min_value=1, max_value=30),
            TemplateParameter("analysis_mode", ParameterType.CHOICE, "Analysis type",
                            default="embedding", choices=["embedding", "segmentation", "classification"]),
        ],
        workflow_skeleton={
            "name": "video-analysis",
            "source": {"type": "UPLOADED", "path": "."},
            "execution_mode": "IN_PROCESS",
            "steps": [
                {"name": "extract_frames", "entrypoint": "steps/extract.py:ExtractFrames",
                 "config": {"bucket": "{{input_bucket}}", "fps": "{{fps}}"}},
                {"name": "analyze", "entrypoint": "steps/analyze.py:AnalyzeFrames", "depends_on": ["extract_frames"],
                 "config": {"mode": "{{analysis_mode}}", "output_bucket": "{{output_bucket}}"}},
            ],
        },
    )


def _custom_pipeline_template() -> PipelineTemplate:
    return PipelineTemplate(
        template_id="custom_pipeline",
        name="Custom Pipeline",
        description="Empty scaffold for user-defined workflows",
        tags=["custom", "scaffold"],
        parameters=[
            TemplateParameter("pipeline_name", ParameterType.STRING, "Name for the pipeline"),
            TemplateParameter("execution_mode", ParameterType.CHOICE, "Execution mode",
                            default="IN_PROCESS", choices=["IN_PROCESS", "SUBPROCESS", "SLURM", "K8S"]),
            TemplateParameter("source_path", ParameterType.PATH, "Path to step scripts", default="."),
        ],
        workflow_skeleton={
            "name": "{{pipeline_name}}",
            "source": {"type": "LOCAL_PATH", "path": "{{source_path}}"},
            "execution_mode": "{{execution_mode}}",
            "steps": [],
        },
    )


class TemplateRegistry:
    """Discover and instantiate pipeline templates.

    Usage::

        registry = TemplateRegistry()
        templates = registry.list_templates()

        template = registry.get("image_embedding")
        workflow = template.render({
            "input_bucket": "user1-images",
            "output_bucket": "user1-results",
        })
    """

    def __init__(self, include_builtins: bool = True) -> None:
        self._templates: Dict[str, PipelineTemplate] = {}
        if include_builtins:
            self._register_builtins()

    def _register_builtins(self) -> None:
        for factory in [
            _image_embedding_template,
            _text_analysis_template,
            _defect_detection_template,
            _video_analysis_template,
            _custom_pipeline_template,
        ]:
            t = factory()
            self._templates[t.template_id] = t

    def register(self, template: PipelineTemplate) -> None:
        self._templates[template.template_id] = template

    def unregister(self, template_id: str) -> bool:
        return self._templates.pop(template_id, None) is not None

    def get(self, template_id: str) -> Optional[PipelineTemplate]:
        return self._templates.get(template_id)

    def list_templates(self, tag: Optional[str] = None) -> List[PipelineTemplate]:
        templates = list(self._templates.values())
        if tag:
            templates = [t for t in templates if tag in t.tags]
        return sorted(templates, key=lambda t: t.template_id)

    def search(self, query: str) -> List[PipelineTemplate]:
        query_lower = query.lower()
        return [
            t for t in self._templates.values()
            if query_lower in t.name.lower()
            or query_lower in t.description.lower()
            or any(query_lower in tag for tag in t.tags)
        ]

    def render(self, template_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        t = self.get(template_id)
        if not t:
            raise KeyError(f"Template not found: {template_id}")
        return t.render(params)
