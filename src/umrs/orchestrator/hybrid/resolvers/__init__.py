"""
ModelRefResolver subpackage — model reference resolution (U-1121 – U-1124).

Re-exports all public names so ``from .resolvers import …`` keeps working.
"""

from .base import (
    IncompatibleAuthorityError,
    ModelRefResolver,
    ResolutionError,
    ResolvedModel,
)
from .filesystem import FilesystemResolver
from .huggingface import HuggingFaceResolver
from .pvc import PVCResolver
from .registry import ResolverRegistry
from .s3 import S3Resolver

__all__ = [
    "FilesystemResolver",
    "HuggingFaceResolver",
    "IncompatibleAuthorityError",
    "ModelRefResolver",
    "PVCResolver",
    "ResolutionError",
    "ResolvedModel",
    "ResolverRegistry",
    "S3Resolver",
]
