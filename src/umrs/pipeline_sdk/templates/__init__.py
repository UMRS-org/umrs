"""Built-in step templates for the UMRS Pipeline SDK.

These templates let users compose a multimodal *perception → reasoning*
pipeline in ``workflow.json`` without uploading any Python code:

.. code-block:: json

   {
     "name": "scientific-figure-understanding",
     "version": "0.1",
     "steps": [
       {
         "name": "vision_dense",
         "entrypoint": "umrs.pipeline_sdk.templates.vision_step:VisionStep",
         "config": {"mode": "dense"}
       },
       {
         "name": "vision_ocr",
         "entrypoint": "umrs.pipeline_sdk.templates.vision_step:VisionStep",
         "config": {"mode": "ocr"}
       },
       {
         "name": "reason",
         "entrypoint": "umrs.pipeline_sdk.templates.llm_step:LLMStep",
         "depends_on": ["vision_dense", "vision_ocr"],
         "config": {
           "model": "rivault-llm-static",
           "system_prompt": "You are a scientific figure analyst.",
           "prompt_template": "Describe the figure: {{ steps.vision_dense.dense_description }}\\nText in figure: {{ steps.vision_ocr.ocr_text }}",
           "output_schema": {"type": "object", "properties": {"finding": {"type": "string"}}}
         }
       }
     ]
   }

Each step uses the ``VisionPerception`` envelope
(``umrs.schemas.vision``) as the I/O contract.
"""

from .vision_step import VisionStep
from .llm_step import LLMStep
from .ocr_step import OCRStep

# Re-export legacy B7 pipeline-template registry symbols so existing
# imports of ``umrs.pipeline_sdk.templates`` keep working after this
# module became a package (was a single file before).
from .._pipeline_templates_legacy import (  # noqa: E402
    ParameterType,
    PipelineTemplate,
    TemplateParameter,
    TemplateRegistry,
)

__all__ = [
    "VisionStep",
    "LLMStep",
    "OCRStep",
    "ParameterType",
    "PipelineTemplate",
    "TemplateParameter",
    "TemplateRegistry",
]
